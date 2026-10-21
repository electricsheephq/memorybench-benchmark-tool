import { writeFileSync } from "node:fs"
import { basename, dirname, resolve } from "node:path"
import type { Benchmark } from "../types/benchmark"
import { CheckpointManager } from "./checkpoint"
import { datasetIdentity, runIdentities } from "./identities"
import { generateReport } from "./phases/report"

// The optional byte changes only the synthetic dataset, never answers or verdicts.
export async function emitIdentityFixture(output: string, byte = "A"): Promise<void> {
  if (!/^[AB]$/.test(byte)) throw new Error("Dataset byte must be A or B")
  const bytes = Buffer.from(JSON.stringify({ byte, questions: ["one_abs", "two"] }))
  const questions = JSON.parse(bytes.toString()).questions.map((questionId: string) => ({
    questionId,
    question: "Synthetic question",
    groundTruth: "Synthetic answer",
    questionType: questionId.endsWith("_abs") ? "single-session-user" : "knowledge-update",
    haystackSessionIds: [],
  }))
  const benchmark: Benchmark = {
    name: "longmemeval",
    ...datasetIdentity("./synthetic.json", bytes),
    async load() {},
    getQuestions: () => questions,
    getHaystackSessions: () => [],
    getGroundTruth: () => "Synthetic answer",
    getQuestionTypes: () => ({}),
  }
  const destination = resolve(output)
  const manager = new CheckpointManager(dirname(destination))
  const checkpoint = manager.create(
    basename(destination),
    "fixture",
    benchmark.name,
    "fixture-judge",
    "fixture-reader",
    {
      identities: runIdentities(benchmark, { name: "fixture" }, "openai:fixture-judge"),
      targetQuestionIds: questions.map((q: { questionId: string }) => q.questionId),
    }
  )
  checkpoint.recallConfig = "OFF"
  for (const q of questions) {
    manager.initQuestion(checkpoint, q.questionId, q.questionId, q)
    for (const phase of ["ingest", "indexing", "search", "answer", "evaluate"] as const) {
      manager.updatePhase(checkpoint, q.questionId, phase, { status: "completed" })
    }
    manager.updatePhase(checkpoint, q.questionId, "search", {
      recallHealth: {
        coverage: { fts: "ok" },
        degraded: false,
        degradedReason: null,
        timeout: false,
      },
    })
    manager.updatePhase(checkpoint, q.questionId, "answer", { hypothesis: q.groundTruth })
    manager.updatePhase(checkpoint, q.questionId, "evaluate", { score: 1, label: "correct" })
  }
  manager.updateStatus(checkpoint, "completed")
  await manager.flush(checkpoint.runId)
  writeFileSync(
    resolve(destination, "report.json"),
    JSON.stringify(generateReport(benchmark, checkpoint), null, 2)
  )
}
