import type { UnifiedQuestion, UnifiedSession, QuestionTypeRegistry } from "./unified"

export interface BenchmarkConfig {
  dataPath?: string
}

export interface QuestionFilter {
  /** Filter by raw question type ids (benchmark-specific) */
  questionTypes?: string[]
  limit?: number
  offset?: number
}

export interface DatasetIdentity {
  datasetSha256: string
  datasetPath: string
}

export interface Benchmark extends Partial<DatasetIdentity> {
  name: string
  load(config?: BenchmarkConfig): Promise<void>
  getQuestions(filter?: QuestionFilter): UnifiedQuestion[]
  getHaystackSessions(questionId: string): UnifiedSession[]
  getGroundTruth(questionId: string): string
  getQuestionTypes(): QuestionTypeRegistry
}

export type BenchmarkName = "locomo" | "longmemeval" | "convomem"
