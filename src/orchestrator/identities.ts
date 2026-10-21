import { createHash } from "node:crypto"
import { isAbsolute, win32 } from "node:path"
import type { Benchmark, DatasetIdentity } from "../types/benchmark"
import type { RunCheckpoint, RunIdentities } from "../types/checkpoint"
import type { Provider } from "../types/provider"
import { buildJudgePrompt, SCORING_VERSION } from "../judges/base"
import { LOCOMO_QUESTION_TYPES } from "../benchmarks/locomo"
import { LONGMEMEVAL_QUESTION_TYPES } from "../benchmarks/longmemeval"
import { CONVOMEM_QUESTION_TYPES } from "../benchmarks/convomem"
import { buildAnswerPrompt } from "./phases/answer"

export function relativeDatasetPath(datasetPath: string): string {
  if (isAbsolute(datasetPath) || win32.isAbsolute(datasetPath)) {
    throw new Error("datasetPath must be a relative label")
  }
  return datasetPath
}

export function datasetIdentity(datasetPath: string, bytes: Uint8Array): DatasetIdentity {
  return {
    datasetPath: relativeDatasetPath(datasetPath),
    datasetSha256: createHash("sha256").update(bytes).digest("hex"),
  }
}

export function runIdentities(
  benchmark: Benchmark,
  provider: Pick<Provider, "name" | "prompts">,
  judge: string
): RunIdentities {
  if (!benchmark.datasetPath || !benchmark.datasetSha256)
    throw new Error("Dataset identity not loaded")
  const override = provider.prompts?.judgePrompt
  // Render every registered type, its alias and the default fallback with fixed
  // inputs. This covers the wrapper and routing, independent of function source
  // formatting/transpilation or closed-over provider prompt text.
  const types = [
    ...new Set([
      "default",
      ...[LOCOMO_QUESTION_TYPES, LONGMEMEVAL_QUESTION_TYPES, CONVOMEM_QUESTION_TYPES].flatMap(
        (registry) => Object.values(registry).flatMap(({ id, alias }) => [id, alias])
      ),
    ]),
  ].sort()
  const judgePrompts = types.map((questionType) => [
    questionType,
    buildJudgePrompt({
      question: "QUESTION",
      groundTruth: "GROUND_TRUTH",
      hypothesis: "HYPOTHESIS",
      questionType,
      providerPrompts: provider.prompts,
    }),
  ])
  const readerPrompts = [undefined, "2000-01-01"].map((date) =>
    buildAnswerPrompt("QUESTION", [{ content: "CONTEXT" }], date, provider)
  )
  return {
    datasetPath: benchmark.datasetPath,
    datasetSha256: benchmark.datasetSha256,
    questionSetSha256: questionSetSha256(benchmark.getQuestions().map((q) => q.questionId)),
    promptVersion: `sha256:${hashJson(judgePrompts)}`,
    readerPromptVersion: `sha256:${hashJson(readerPrompts)}`,
    scoringPath: `${judge}/${override ? `provider:${provider.name}` : "default"}/${SCORING_VERSION}`,
    answerPresentationMode: "raw_json_v1",
  }
}

function hashJson(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex")
}

export function questionSetSha256(orderedIds: string[]): string {
  return hashJson(orderedIds)
}

export function assertResumeIdentities(checkpoint: RunCheckpoint, current: RunIdentities): void {
  for (const key of [
    "datasetSha256",
    "questionSetSha256",
    "promptVersion",
    "readerPromptVersion",
    "scoringPath",
  ] as const) {
    if (checkpoint[key] !== undefined && checkpoint[key] !== current[key]) {
      throw new Error(`Resume refused: identity mismatch: ${key}`)
    }
  }
}
