import { expect, test } from "bun:test"
import { mkdtempSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { CheckpointManager } from "./checkpoint"
import { runSearchPhase } from "./phases/search"
import { generateReport } from "./phases/report"
import { HermesLcmProvider } from "../providers/hermes-lcm"
import type { Benchmark } from "../types/benchmark"
import type { Provider, RecallHealth } from "../types/provider"
import type { RunCheckpoint } from "../types/checkpoint"

const health: RecallHealth = {
  degraded: true,
  degradedReason: "query unavailable",
  timeout: true,
  coverage: { fts: "ok", summary: "none" },
}
const question = {
  questionId: "q_abs",
  question: "fixture?",
  questionType: "single-session-user",
  groundTruth: "fixture",
  haystackSessionIds: [],
}
const benchmark = {
  getQuestions: () => [question],
  getQuestionTypes: () => ({}),
} as unknown as Benchmark
function setup() {
  const root = mkdtempSync(join(process.env.TMPDIR!, "recall-health-"))
  const manager = new CheckpointManager(root)
  const checkpoint = {
    runId: "fixture",
    dataSourceRunId: "fixture",
    provider: "stub",
    benchmark: "longmemeval",
    judge: "stub",
    answeringModel: "stub",
    status: "completed",
    questions: {
      q_abs: {
        ...question,
        containerTag: "q_abs-fixture",
        phases: {
          ingest: { status: "completed", completedSessions: [] },
          indexing: { status: "completed" },
          search: { status: "pending" },
          answer: { status: "completed", hypothesis: "fixture" },
          evaluate: { status: "completed", label: "correct", score: 1 },
        },
      },
    },
  } as RunCheckpoint
  return { root, manager, checkpoint }
}

test("health-aware search persists identical health in result, checkpoint and report", async () => {
  const { manager, checkpoint } = setup()
  const provider = {
    recallConfig: "OFF",
    search: async () => {
      throw Error("legacy called")
    },
    searchWithHealth: async () => ({ results: [{ content: "fixture" }], health }),
  } as unknown as Provider
  await runSearchPhase(provider, benchmark, checkpoint, manager)
  await manager.flush()
  const search = checkpoint.questions.q_abs!.phases.search
  expect(JSON.parse(readFileSync(search.resultFile!, "utf8")).recallHealth).toEqual(health)
  expect(manager.load("fixture")!.questions.q_abs!.phases.search.recallHealth).toEqual(health)
  const report = generateReport(benchmark, checkpoint)
  expect(report.recallConfig).toBe("OFF")
  expect(report.evaluations[0]!.recallHealth).toEqual(health)
})

test("legacy provider and old checkpoint expose explicit missing health without score changes", async () => {
  const { manager, checkpoint } = setup()
  const before = generateReport(benchmark, checkpoint)
  await runSearchPhase(
    { search: async () => [] } as unknown as Provider,
    benchmark,
    checkpoint,
    manager
  )
  await manager.flush()
  expect(checkpoint.questions.q_abs!.phases.search.recallHealth).toBeNull()
  const report = generateReport(benchmark, checkpoint)
  expect(report.recallConfig).toBe("UNKNOWN")
  expect(report.evaluations[0]!.recallHealth).toBeNull()
  expect(report.summary).toEqual(before.summary)
})

test("checkpoint copies preserve config only when preserving search", async () => {
  const { manager, checkpoint } = setup()
  checkpoint.recallConfig = "LOCAL"
  manager.save(checkpoint)
  await manager.flush()
  expect(manager.copyCheckpoint("fixture", "answer-copy", "answer").recallConfig).toBe("LOCAL")
  expect(manager.copyCheckpoint("fixture", "search-copy", "search").recallConfig).toBeUndefined()
  await manager.flush()
})

test("Hermes returns request-local health and leaves quota coverage absent", async () => {
  const { root } = setup()
  const provider = new HermesLcmProvider()
  const state = provider as any
  state.workdir = root
  state.handles.set("ordinary", {
    request: async () => ({
      ok: true,
      results: ["hit"],
      degraded: true,
      degraded_reason: health.degradedReason,
      timeout: true,
      provenance: { coverage: health.coverage },
    }),
  })
  state.handles.set("quota", { request: async () => ({ ok: true, results: [] }) })
  const [ordinary, quota] = await Promise.all([
    provider.searchWithHealth("q", { containerTag: "ordinary" }),
    provider.searchWithHealth("q", { containerTag: "quota" }),
  ])
  expect(ordinary.health).toEqual(health)
  expect(quota.health.coverage).toBeNull()
  expect(await provider.search("q", { containerTag: "ordinary" })).toEqual(["hit"])
})

test.each([
  [false, "voyage", "OFF"],
  [true, "fastembed", "LOCAL"],
  [true, "voyage", "VOYAGE-TRACKER"],
  [true, "stub", "UNKNOWN"],
])("Hermes initialized settings %s/%s record %s", async (enabled, providerName, expected) => {
  const provider = new HermesLcmProvider()
  ;(provider as any).spawnHandle = () => ({
    request: async () => ({
      ok: true,
      embeddings_enabled: enabled,
      provider: providerName,
    }),
    close: async () => {},
  })
  const previous = process.env.HERMES_LCM_PYTHON
  process.env.HERMES_LCM_PYTHON = process.execPath
  try {
    await provider.initialize({ apiKey: "" })
  } finally {
    if (previous === undefined) delete process.env.HERMES_LCM_PYTHON
    else process.env.HERMES_LCM_PYTHON = previous
  }
  expect(provider.recallConfig).toBe(expected)
})
