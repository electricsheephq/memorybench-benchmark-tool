import type { UnifiedSession } from "./unified"
import type { ProviderPrompts } from "./prompts"
import type { ConcurrencyConfig } from "./concurrency"

export interface ProviderConfig {
  apiKey: string
  baseUrl?: string
  [key: string]: unknown
}

export interface IngestOptions {
  containerTag: string
  metadata?: Record<string, unknown>
}

export interface SearchOptions {
  containerTag: string
  limit?: number
  threshold?: number
}

export interface IngestResult {
  documentIds: string[]
  taskIds?: string[]
}

export interface IndexingProgress {
  completedIds: string[]
  failedIds: string[]
  total: number
}

export type IndexingProgressCallback = (progress: IndexingProgress) => void

export type RecallConfig = "OFF" | "LOCAL" | "VOYAGE-TRACKER" | "UNKNOWN"

export interface RecallHealth {
  degraded: boolean
  degradedReason: string | null
  timeout: boolean
  coverage: Record<string, string> | null
}

export interface Provider {
  recallConfig?: RecallConfig
  name: string
  prompts?: ProviderPrompts
  concurrency?: ConcurrencyConfig
  initialize(config: ProviderConfig): Promise<void>
  ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult>
  awaitIndexing(
    result: IngestResult,
    containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void>
  search(query: string, options: SearchOptions): Promise<unknown[]>
  searchWithHealth?(
    query: string,
    options: SearchOptions
  ): Promise<{ results: unknown[]; health: RecallHealth }>
  clear(containerTag: string): Promise<void>
}

export type ProviderName =
  | "supermemory"
  | "mem0"
  | "zep"
  | "filesystem"
  | "rag"
  | "cortex"
  | "hermes-lcm"
