import { createHash } from "node:crypto"
import { isAbsolute, win32 } from "node:path"
import type { Benchmark, DatasetIdentity } from "../types/benchmark"
import type { RunCheckpoint, RunIdentities } from "../types/checkpoint"
import type { Provider } from "../types/provider"
import { buildJudgePrompt, SCORING_VERSION } from "../judges/base"
import { LOCOMO_QUESTION_TYPES } from "../benchmarks/locomo"
import { LONGMEMEVAL_QUESTION_TYPES } from "../benchmarks/longmemeval"
import { CONVOMEM_QUESTION_TYPES } from "../benchmarks/convomem"
import { resolveModel } from "../utils/models"
import { buildAnswerPrompt } from "./phases/answer"
import { cliLlmProvenance } from "../utils/cli-llm"

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
  // Fixed, renderable input for every registered provider; renderer errors must
  // propagate rather than silently substituting a different prompt identity.
  const context = {
    content: "CONTEXT",
    score: 1,
    vectorScore: 1,
    bm25Score: 1,
    sessionId: "SESSION",
    chunkIndex: 0,
    date: "2000-01-01",
    memory: "MEMORY",
    fact: "FACT",
    valid_at: "2000-01-01",
    metadata: {
      date: "2000-01-01",
      timestamp: "2000-01-01",
      event_date: "2000-01-01",
      session_date: "2000-01-01",
      speaker: "SPEAKER",
      source_session_id: "SESSION",
      temporalContext: { documentDate: "2000-01-01", eventDate: "2000-01-01" },
    },
    chunks: [{ content: "CHUNK", position: 0 }],
    chunk: "CHUNK",
    subject: "SUBJECT",
    predicate: "PREDICATE",
    object: "OBJECT",
  }
  // Shared fixed branch matrix, rendered separately through each provider:
  // empty/full/sparse context; scalar/list question; absent/present date.
  // Full hits cover Cortex triples, Zep edges and Supermemory array chunks;
  // sparse hits cover text fallbacks, Zep nodes and singular/no chunks.
  const contexts = [
    [],
    [context],
    [{ content: "CONTENT", memory: "MEMORY", fact: "FACT", chunk: "CHUNK" }],
    [{ _type: "node", name: "ENTITY", summary: "SUMMARY" }],
    [{ text: "TEXT", metadata: {} }],
    [{ fact: "FACT", metadata: { date: "2000-01-01" } }],
    [{ content: "CONTENT", metadata: { timestamp: "2000-01-01" } }],
    [
      {
        subject: ["SUBJECT", null],
        predicate: 1,
        object: { value: true },
        metadata: { session_date: "2000-01-01" },
      },
    ],
    [{ memory: "MEMORY", metadata: { temporalContext: { eventDate: ["2000-01-01"] } } }],
  ]
  const readerPrompts = ["QUESTION", "List all items"].flatMap((question) =>
    contexts.flatMap((hits) =>
      [undefined, "2000-01-01"].map((date) =>
        buildAnswerPrompt(
          question,
          hits.map((hit) => ({
            score: 1,
            vectorScore: 1,
            bm25Score: 1,
            sessionId: "SESSION",
            ...hit,
          })),
          date,
          provider
        )
      )
    )
  )
  const questions = benchmark.getQuestions()
  const normalizedInputs = [...questions]
    .sort((a, b) => a.questionId.localeCompare(b.questionId))
    .map((question) => ({ question, sessions: benchmark.getHaystackSessions(question.questionId) }))
  const cliJudge = judge.startsWith("cli:") ? cliLlmProvenance("judge") : null
  return {
    datasetPath: benchmark.datasetPath,
    datasetSha256: benchmark.datasetSha256,
    questionSetSha256: questionSetSha256(questions.map((q) => q.questionId)),
    normalizedInputsSha256: hashJson(normalizedInputs),
    loaderOrderSha256: questionSetSha256(questions.map((q) => q.questionId)),
    promptVersion: `sha256:${hashJson(judgePrompts)}`,
    readerPromptVersion: `sha256:${hashJson(readerPrompts)}`,
    scoringPath: `${judge}/${override ? `provider:${provider.name}` : "default"}/${SCORING_VERSION}${cliJudge ? `/cli-sha256:${hashJson(cliJudge)}` : ""}`,
    answerPresentationMode: "raw_json_v1",
  }
}

function hashJson(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex")
}

export function questionSetSha256(orderedIds: string[]): string {
  return hashJson(orderedIds)
}

export function assertResumeIdentities(
  checkpoint: RunCheckpoint,
  current: RunIdentities,
  phases: readonly string[] = ["answer", "evaluate"]
): void {
  const answers = phases.includes("answer")
  const scores = phases.includes("evaluate")
  for (const key of [
    "datasetSha256",
    "questionSetSha256",
    "normalizedInputsSha256",
    "loaderOrderSha256",
    ...(answers ? ["readerPromptVersion" as const] : []),
    ...(scores ? ["promptVersion" as const, "scoringPath" as const] : []),
    ...(answers || scores ? ["answerPresentationMode" as const] : []),
  ] as const) {
    let expected = current[key]
    if (key === "scoringPath" && /^(openai|anthropic|google|cli)\//.test(checkpoint[key] ?? "")) {
      // Round-2 identities lack the model. Compare the old path format, then
      // use the recorded judge (or CLI provenance when known) to check the model.
      const [judge, ...path] = expected.split("/")
      const [provider, ...model] = judge!.split(":")
      expected = `${provider}/${path.join("/")}`
      const storedModel =
        provider === "cli" ? checkpoint.judgeProvenance?.model : resolveModel(checkpoint.judge).id
      const currentModel = provider === "cli" ? model.slice(1).join(":") : model.join(":")
      const storedTransport = checkpoint.judgeProvenance?.transport
      if (
        (storedModel && storedModel !== currentModel) ||
        (provider === "cli" && storedTransport && storedTransport !== `${model[0]}-cli`)
      ) {
        throw new Error("Resume refused: identity mismatch: scoringPath")
      }
    }
    if (checkpoint[key] !== undefined && checkpoint[key] !== expected) {
      throw new Error(`Resume refused: identity mismatch: ${key}`)
    }
  }
}
