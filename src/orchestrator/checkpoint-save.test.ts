import { afterEach, describe, expect, spyOn, test } from "bun:test"
import * as fs from "fs"
import { join } from "path"
import { tmpdir } from "os"
import { CheckpointManager } from "./checkpoint"
import { Orchestrator } from "./index"
import { HermesLcmProvider } from "../providers/hermes-lcm"
import { LoCoMoBenchmark } from "../benchmarks/locomo"
import type { RunCheckpoint } from "../types/checkpoint"

const roots: string[] = []
const restore: Array<() => void> = []

afterEach(() => {
  for (const undo of restore.splice(0).reverse()) undo()
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})

function fixture() {
  const root = fs.mkdtempSync(join(tmpdir(), "checkpoint-save-"))
  roots.push(root)
  const manager = new CheckpointManager(root)
  const checkpoint = manager.create("fixture", "hermes-lcm", "locomo", "gpt-4o", "gpt-4o")
  return { manager, checkpoint }
}

describe("coalesced checkpoint saves", () => {
  test("50 large saves coalesce, persist the final state, and let a timer run", async () => {
    const { manager, checkpoint } = fixture()
    manager.initQuestion(checkpoint, "q", "fixture-container", {
      question: "x".repeat(5 * 1024 * 1024), groundTruth: "fixture", questionType: "fixture",
    })
    expect(JSON.stringify(checkpoint).length).toBeGreaterThanOrEqual(5 * 1024 * 1024)
    const rename = fs.renameSync
    const timerStates: boolean[] = []
    let timerFired = false
    const writes = spyOn(fs, "renameSync").mockImplementation((source, target) => {
      rename(source, target)
      timerStates.push(timerFired)
      // Simulate a new save arriving while the first write is in flight.
      if (timerStates.length === 1) {
        checkpoint.limit = 50
        manager.save(checkpoint)
      }
    })
    restore.push(() => writes.mockRestore())
    for (let i = 0; i < 50; i++) {
      checkpoint.limit = i
      manager.save(checkpoint)
    }
    const timer = new Promise<void>((resolve) => setTimeout(() => {
      timerFired = true
      resolve()
    }, 0))
    await manager.flush(checkpoint.runId)
    await timer
    expect(writes.mock.calls.length).toBeLessThanOrEqual(2)
    expect(timerStates.at(-1)).toBe(true)
    expect(manager.load(checkpoint.runId)).toEqual(checkpoint)
    expect(fs.readFileSync(manager.getCheckpointPath(checkpoint.runId), "utf8"))
      .toBe(JSON.stringify(manager.load(checkpoint.runId)))
  })

  test("flush waits for the pending dirty write, including a replacement checkpoint", async () => {
    const { manager, checkpoint } = fixture()
    const writer = manager as unknown as { _performSave(checkpoint: RunCheckpoint): Promise<void> }
    const performSave = writer._performSave.bind(manager)
    let signalWritten!: () => void
    const firstWritten = new Promise<void>((resolve) => { signalWritten = resolve })
    let release!: () => void
    const gate = new Promise<void>((resolve) => { release = resolve })
    let count = 0
    const writes = spyOn(writer, "_performSave").mockImplementation(async (latest) => {
      await performSave(latest)
      if (++count === 1) {
        signalWritten()
        await gate
      }
    })
    restore.push(() => writes.mockRestore())
    await firstWritten
    const final = { ...checkpoint, limit: 123 }
    manager.save(final)
    let flushed = false
    const flushing = manager.flush(checkpoint.runId).then(() => { flushed = true })
    await new Promise<void>((resolve) => setImmediate(resolve))
    expect(flushed).toBe(false)
    expect(manager.load(checkpoint.runId)?.limit).toBeUndefined()
    release()
    await flushing
    expect(count).toBe(2)
    expect(manager.load(checkpoint.runId)).toEqual(final)
  })
})

describe("orchestrator provider cleanup", () => {
  for (const fails of [true, false]) {
    test(`search ${fails ? "failure rejects" : "success completes"} after flushing and closing once`, async () => {
      const { manager, checkpoint } = fixture()
      const question = {
        questionId: "q", question: "fixture?", groundTruth: "fixture",
        questionType: "fixture", haystackSessionIds: [],
      }
      manager.initQuestion(checkpoint, "q", "fixture-container", question)
      checkpoint.targetQuestionIds = ["q"]
      checkpoint.questions.q!.phases.indexing.status = "completed"
      manager.save(checkpoint)
      await manager.flush(checkpoint.runId)
      const load = spyOn(LoCoMoBenchmark.prototype, "load").mockResolvedValue(undefined)
      const questions = spyOn(LoCoMoBenchmark.prototype, "getQuestions").mockReturnValue([question])
      const initialize = spyOn(HermesLcmProvider.prototype, "initialize").mockResolvedValue(undefined)
      const search = spyOn(HermesLcmProvider.prototype, "search").mockImplementation(async () => {
        if (fails) throw new Error("fixture search failure")
        return []
      })
      const close = spyOn(HermesLcmProvider.prototype, "close").mockImplementation(() => {
        const saved = manager.load(checkpoint.runId)!
        expect(saved.questions.q!.phases.search.status).toBe(fails ? "failed" : "completed")
        expect(saved.status).toBe(fails ? "running" : "completed")
      })
      for (const mock of [load, questions, initialize, search, close]) {
        restore.push(() => mock.mockRestore())
      }
      const orchestrator = new Orchestrator()
      ;(orchestrator as unknown as { checkpointManager: CheckpointManager }).checkpointManager = manager
      const running = orchestrator.run({
        provider: "hermes-lcm", benchmark: "locomo", judgeModel: "gpt-4o",
        runId: checkpoint.runId, phases: ["search"],
      })
      if (fails) await expect(running).rejects.toThrow("fixture search failure")
      else await running
      expect(search).toHaveBeenCalledTimes(1)
      expect(close).toHaveBeenCalledTimes(1)
    })
  }
})
