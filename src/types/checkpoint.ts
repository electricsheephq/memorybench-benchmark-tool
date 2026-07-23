import type { SearchResult, RetrievalMetrics } from "./unified"
import type { IngestResult } from "./provider"
import type { ConcurrencyConfig } from "./concurrency"

export type PhaseStatus = "pending" | "in_progress" | "completed" | "failed"

export type PhaseId = "ingest" | "indexing" | "search" | "answer" | "evaluate" | "report"

export const PHASE_ORDER: PhaseId[] = [
  "ingest",
  "indexing",
  "search",
  "answer",
  "evaluate",
  "report",
]

export function getPhasesFromPhase(fromPhase: PhaseId): PhaseId[] {
  const startIndex = PHASE_ORDER.indexOf(fromPhase)
  if (startIndex === -1) return PHASE_ORDER
  return PHASE_ORDER.slice(startIndex)
}

export interface IngestPhaseCheckpoint {
  status: PhaseStatus
  completedSessions: string[]
  ingestResult?: IngestResult
  startedAt?: string
  completedAt?: string
  durationMs?: number
  error?: string
}

export interface IndexingPhaseCheckpoint {
  status: PhaseStatus
  completedIds?: string[]
  failedIds?: string[]
  startedAt?: string
  completedAt?: string
  durationMs?: number
  error?: string
}

export interface SearchPhaseCheckpoint {
  status: PhaseStatus
  resultFile?: string
  results?: SearchResult[]
  resultCount?: number
  startedAt?: string
  completedAt?: string
  durationMs?: number
  error?: string
}

export interface AnswerPhaseCheckpoint {
  status: PhaseStatus
  hypothesis?: string
  promptTokens?: number
  basePromptTokens?: number
  contextTokens?: number
  answerPresentation?: import("../prompts/evidence-cards").EvidenceCardPresentation
  deterministicOperation?: {
    status: "computed" | "fallback" | "not_attempted"
    reason?: string
    trace?: import("../orchestrator/deterministic-operations").DeterministicOperationTrace
  }
  llmCall?: import("../utils/cli-llm").CliCallTelemetry
  llmCalls?: import("../utils/cli-llm").CliCallTelemetry[]
  startedAt?: string
  completedAt?: string
  durationMs?: number
  error?: string
}

export interface EvaluatePhaseCheckpoint {
  status: PhaseStatus
  label?: "correct" | "incorrect"
  score?: number
  explanation?: string
  retrievalMetrics?: RetrievalMetrics
  llmCall?: import("../utils/cli-llm").CliCallTelemetry
  llmCalls?: import("../utils/cli-llm").CliCallTelemetry[]
  startedAt?: string
  completedAt?: string
  durationMs?: number
  error?: string
}

export interface SessionMetadata {
  sessionId: string
  date?: string
  messageCount: number
}

export interface QuestionCheckpoint {
  questionId: string
  containerTag: string
  question: string
  groundTruth: string
  questionType: string
  questionDate?: string
  sessions?: SessionMetadata[]
  phases: {
    ingest: IngestPhaseCheckpoint
    indexing: IndexingPhaseCheckpoint
    search: SearchPhaseCheckpoint
    answer: AnswerPhaseCheckpoint
    evaluate: EvaluatePhaseCheckpoint
  }
}

export type RunStatus = "initializing" | "running" | "completed" | "failed"

export type SelectionMode = "full" | "sample" | "limit"
export type SampleType = "consecutive" | "random"

export interface SamplingConfig {
  mode: SelectionMode
  sampleType?: SampleType
  perCategory?: number
  limit?: number
}

export interface LlmExecutionProvenance {
  transport: "ai-sdk" | "codex-cli" | "claude-cli" | "mixed"
  transportVersion?: string
  model: string
  modelExplicit: boolean
  configuredModel: string
  reasoningEffort?: string
  provider?: string
  serviceTier?: string
  isolated?: boolean
  modelPinSource?: "explicit-cli-argv" | "un-pinned" | "mixed"
  eventUsageCapture?: "codex-jsonl" | "unavailable"
  eventModelField?: "not-emitted-by-codex-jsonl" | "unavailable"
  tokenizerModel?: string
  callCount?: number
  retryCount?: number
  executionIdentityCount?: number
  mixedExecutionIdentity?: boolean
  callLedgerComplete?: boolean
}

export interface CliLedgerSummary {
  callCount: number
  retryCount: number
  executionIdentityCount: number
  mixedExecutionIdentity: boolean
  callLedgerComplete: boolean
}

export interface RunCheckpoint {
  runId: string
  dataSourceRunId: string
  status: RunStatus
  provider: string
  benchmark: string
  judge: string
  answeringModel: string
  answerPresentationMode?: import("../prompts/evidence-cards").AnswerPresentationMode
  answererProvenance?: LlmExecutionProvenance
  judgeProvenance?: LlmExecutionProvenance
  createdAt: string
  updatedAt: string
  limit?: number
  sampling?: SamplingConfig
  targetQuestionIds?: string[]
  concurrency?: ConcurrencyConfig
  questions: Record<string, QuestionCheckpoint>
}
