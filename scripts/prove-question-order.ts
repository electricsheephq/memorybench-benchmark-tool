// Offline base/candidate comparison: outputs ids and hashes only, never corpus text.
import { mock } from "bun:test"
import { createHash } from "node:crypto"
import { readFileSync, writeFileSync } from "node:fs"
import { relative, resolve } from "node:path"

const [sourceArg, dataArg, outputArg] = process.argv.slice(2)
if (!sourceArg || !dataArg || !outputArg) {
  throw new Error("Usage: bun scripts/prove-question-order.ts SOURCE DATA OUTPUT")
}
const source = resolve(sourceArg)
const { LongMemEvalBenchmark } = await import(source + "/src/benchmarks/longmemeval/index.ts")
const benchmark = new LongMemEvalBenchmark()
await benchmark.load({ dataPath: relative(process.cwd(), resolve(dataArg)) })
benchmark.load = async () => {}
mock.module(source + "/src/benchmarks/index.ts", () => ({ createBenchmark: () => benchmark }))
mock.module(source + "/src/providers/index.ts", () => ({
  createProvider: () => ({ name: "fixture", async initialize() {} }),
}))
mock.module(source + "/src/utils/config.ts", () => ({
  config: {},
  getProviderConfig: () => ({ apiKey: "" }),
  getJudgeConfig: () => ({ apiKey: "" }),
}))
const { Orchestrator } = await import(source + "/src/orchestrator/index.ts")
const rows = []
for (const selection of [
  { runId: "limit-10", limit: 10 },
  { runId: "limit-100", limit: 100 },
  { runId: "sample-1", sampling: { mode: "sample", sampleType: "consecutive", perCategory: 1 } },
  { runId: "sample-3", sampling: { mode: "sample", sampleType: "consecutive", perCategory: 3 } },
  { runId: "full" },
]) {
  const runner = new Orchestrator()
  await runner.run({
    provider: "filesystem",
    benchmark: "longmemeval",
    judgeModel: "gpt-4o",
    answeringModel: "gpt-4o",
    phases: ["report"],
    ...selection,
  })
  const checkpoint = JSON.parse(
    readFileSync(`data/runs/${selection.runId}/checkpoint.json`, "utf8")
  )
  const report = JSON.parse(readFileSync(`data/runs/${selection.runId}/report.json`, "utf8"))
  const executionIds = Object.keys(checkpoint.questions)
  const selectedIds =
    checkpoint.targetQuestionIds ??
    benchmark.getQuestions().map((q: { questionId: string }) => q.questionId)
  const hash = createHash("sha256").update(JSON.stringify(selectedIds)).digest("hex")
  if (
    checkpoint.questionSetSha256 !== undefined &&
    (checkpoint.questionSetSha256 !== hash || report.questionSetSha256 !== hash)
  ) {
    throw new Error("Question selection identity did not match selected ids")
  }
  rows.push({
    config: selection.runId,
    selectedIds,
    executionIds,
    questionSetSha256: hash,
    datasetSha256: benchmark.datasetSha256,
  })
}
writeFileSync(resolve(outputArg), JSON.stringify(rows, null, 2) + "\n")
console.log("Order selection proof complete")
