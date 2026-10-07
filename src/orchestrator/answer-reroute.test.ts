import { afterEach, describe, expect, test } from "bun:test"
import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { tmpdir } from "node:os"
import type { Benchmark } from "../types/benchmark"
import { CheckpointManager } from "./checkpoint"
import { runAnswerPhase } from "./phases/answer"

const originalEnv = { ...process.env }
const tempPaths: string[] = []

afterEach(() => {
  for (const key of Object.keys(process.env)) delete process.env[key]
  Object.assign(process.env, originalEnv)
  for (const path of tempPaths.splice(0)) rmSync(path, { recursive: true, force: true })
})

function fixture(event: unknown) {
  const root = mkdtempSync(join(tmpdir(), "memorybench-answer-reroute-"))
  tempPaths.push(root)
  writeFileSync(
    join(root, "codex"),
    `#!/bin/sh
if [ "$1" = "--version" ]; then
  printf 'codex-cli 1.2.3\\n'
  exit 0
fi
out=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    shift
    out="$1"
  fi
  shift
done
IFS= read -r _prompt || true
printf 'synthetic answer' > "$out"
printf '%s\\n' '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":4}}'
printf '%s\\n' '${JSON.stringify(event).replaceAll("'", "'\\''")}'
exit 0
`,
    { mode: 0o755 }
  )
  process.env.PATH = `${root}:${originalEnv.PATH || ""}`
  process.env.HERMES_MB_LLM_CLI = "codex"
  process.env.HERMES_MB_CODEX_MODEL = "gpt-6.1-sol"
  process.env.HERMES_MB_CODEX_ANSWER_MODEL = "gpt-6.1-sol"

  const manager = new CheckpointManager(join(root, "runs"))
  const checkpoint = manager.create("synthetic-run", "rag", "longmemeval", "gpt-4o", "gpt-4o", {
    concurrency: { default: 1 },
  })
  const questions = ["q1", "q2"].map((questionId) => ({
    questionId,
    question: "Synthetic question?",
    groundTruth: "Synthetic answer",
    questionType: "single-session-user",
    haystackSessionIds: [],
  }))
  for (const question of questions) {
    manager.initQuestion(checkpoint, question.questionId, "synthetic-container", {
      question: question.question,
      groundTruth: question.groundTruth,
      questionType: question.questionType,
    })
    const resultFile = join(manager.getResultsDir(checkpoint.runId), `${question.questionId}.json`)
    writeFileSync(resultFile, JSON.stringify({ results: [{ content: "Synthetic context" }] }))
    checkpoint.questions[question.questionId]!.phases.search = { status: "completed", resultFile }
  }
  const benchmark = {
    name: "longmemeval",
    load: async () => {},
    getQuestions: () => questions,
    getHaystackSessions: () => [],
    getGroundTruth: () => "Synthetic answer",
    getQuestionTypes: () => ({}),
  } satisfies Benchmark
  return { benchmark, checkpoint, manager }
}

describe("answer failure handling", () => {
  test("a reroute rejects after recording the failed answer and stops the next batch", async () => {
    const { benchmark, checkpoint, manager } = fixture({
      type: "item.completed",
      item: { type: "error", message: "model rerouted: gpt-6.1-sol -> other-model-1" },
    })
    try {
      await expect(runAnswerPhase(benchmark, checkpoint, manager)).rejects.toThrow(
        "Answer q1: model rerouted; stopping per RS-ROW1 §6"
      )
    } finally {
      await manager.flush(checkpoint.runId)
    }
    const saved = manager.load(checkpoint.runId)!
    const answer = saved.questions.q1!.phases.answer
    expect(answer.status).toBe("failed")
    expect(answer.hypothesis).toBeUndefined()
    expect(answer.llmCalls).toHaveLength(1)
    expect(answer.llmCalls?.[0]?.attempts).toHaveLength(1)
    expect(answer.llmCalls?.[0]?.attempts[0]).toMatchObject({
      status: "failed",
      errorItemCount: 1,
      reroute: { requested: "gpt-6.1-sol", served: "other-model-1" },
    })
    expect(saved.questions.q2!.phases.answer.status).toBe("pending")
  })

  test("a plain failed turn marks questions failed and continues", async () => {
    const { benchmark, checkpoint, manager } = fixture({
      type: "turn.failed",
      error: { message: "Synthetic turn failure" },
    })
    try {
      await expect(runAnswerPhase(benchmark, checkpoint, manager)).resolves.toBeUndefined()
    } finally {
      await manager.flush(checkpoint.runId)
    }
    const saved = manager.load(checkpoint.runId)!
    for (const id of ["q1", "q2"]) {
      const answer = saved.questions[id]!.phases.answer
      expect(answer.status).toBe("failed")
      expect(answer.llmCalls).toHaveLength(1)
      expect(answer.llmCalls?.[0]?.attempts[0]?.reroute).toBeUndefined()
    }
  })
})
