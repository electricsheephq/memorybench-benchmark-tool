import { createHash } from "node:crypto"
import { isAbsolute, win32 } from "node:path"
import type { Benchmark, DatasetIdentity } from "../types/benchmark"
import type { RunCheckpoint, RunIdentities } from "../types/checkpoint"
import type { Provider } from "../types/provider"
import type { UnifiedSession } from "../types/unified"
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
  judge: string,
  legacyReader = false
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
  const contexts: Record<string, unknown>[][] = [
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
    ...missingFieldInputs(context).map((hit) => [hit]),
    ...missingFieldInputs({ _type: "node", name: "ENTITY", summary: "SUMMARY" }).map((hit) => [
      hit,
    ]),
    ...missingFieldInputs({
      chunks: [
        { content: "LATER", position: 2 },
        { content: "EARLIER", position: 1 },
        { content: "LATER", position: 2 },
      ],
    }).map((hit) => [hit]),
    [context, context],
  ]
  const readerPrompts = [
    "QUESTION",
    "List all items",
    "all items",
    "list items",
    "which items",
    "what were the different items",
    "name every item",
    "name all",
    "what are the items",
  ].flatMap((question) =>
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
  const sessions = new Map<string, Set<string>>()
  const serializedSessions = new WeakMap<UnifiedSession, string>()
  const normalizedInputs = [...questions]
    .sort((a, b) => a.questionId.localeCompare(b.questionId))
    .map((question) => ({
      question,
      sessions: benchmark.getHaystackSessions(question.questionId).map((session) => {
        const content = serializedSessions.get(session) ?? JSON.stringify(session)
        serializedSessions.set(session, content)
        const variants = sessions.get(session.sessionId) ?? new Set<string>()
        variants.add(content)
        sessions.set(session.sessionId, variants)
        return session.sessionId
      }),
    }))
  // Sort ids as before; retain distinct content in first-seen question order.
  // Reuse serialized session JSON so shared objects are read only once, while
  // preserving the exact previous JSON bytes for datasets without conflicting ids.
  const sessionEntries = [...sessions]
    .sort(([a], [b]) => a.localeCompare(b))
    .flatMap(([id, variants]) => [...variants].map((content) => `[${JSON.stringify(id)},${content}]`))
  const normalizedInputsSha256 = createHash("sha256")
    .update(`{"questions":${JSON.stringify(normalizedInputs)},"sessions":[${sessionEntries.join(",")}]}`)
    .digest("hex")
  const cliJudge = judge.startsWith("cli:") ? cliLlmProvenance("judge") : null
  return {
    datasetPath: benchmark.datasetPath,
    datasetSha256: benchmark.datasetSha256,
    questionSetSha256: questionSetSha256(questions.map((q) => q.questionId)),
    normalizedInputsSha256,
    loaderOrderSha256: questionSetSha256(questions.map((q) => q.questionId)),
    promptVersion: `sha256:${hashJson(judgePrompts)}`,
    readerPromptVersion: legacyReader
      ? `sha256:${hashJson([undefined, "2000-01-01"].map((date) => buildAnswerPrompt("QUESTION", [context], date, provider)))}`
      : `rpv2:sha256:${hashJson(readerPrompts)}`,
    scoringPath: `${judge}/${override ? `provider:${provider.name}` : "default"}/${SCORING_VERSION}${cliJudge ? `/cli-sha256:${hashJson(cliJudge)}` : ""}`,
    answerPresentationMode: "raw_json_v1",
  }
}

// Remove each optional field once, including nested metadata and chunk fields.
// Required numeric RAG fields remain renderable in every generated input.
function missingFieldInputs(input: Record<string, unknown>): Record<string, unknown>[] {
  return Object.entries(input).flatMap(([key, value]) => {
    const absent = { ...input }
    delete absent[key]
    const variants = [absent]
    if (value && typeof value === "object") {
      const entries = Array.isArray(value) ? value : [value]
      entries.forEach((entry, index) => {
        if (!entry || typeof entry !== "object") return
        for (const nested of missingFieldInputs(entry)) {
          const replacement = [...entries]
          replacement[index] = nested
          variants.push({ ...input, [key]: Array.isArray(value) ? replacement : nested })
        }
      })
    }
    return ["score", "vectorScore", "bm25Score", "sessionId"].includes(key) ? [] : variants
  })
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
