import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import type { ConcurrencyConfig } from "../../types/concurrency"
import { logger } from "../../utils/logger"
import { config } from "../../utils/config"
import { CORTEX_PROMPTS } from "./prompts"

/** Cortex memory provider for MemoryBench. Uses /api/v1/memories/remember + /api/v1/memories/search. */
export class CortexProvider implements Provider {
  name = "cortex"
  prompts = CORTEX_PROMPTS

  concurrency: ConcurrencyConfig = {
    default: 3,
    ingest: 1,
    search: 5,
  }

  private baseUrl: string = ""
  private apiKey: string = ""
  private ownerId: string = ""
  private entityType: string = "system"
  private sourceAgent: string = "memorybench"

  /** Build default request headers */
  /** Create an AbortSignal that fires after timeoutMs.
   * Reads CORTEX_TIMEOUT_MS env var if set, otherwise defaults to 600_000ms (10 min).
   */
  private timeoutSignal(timeoutMs?: number): AbortSignal {
    if (timeoutMs === undefined) {
      const envMs = process.env.CORTEX_TIMEOUT_MS;
      const parsed = envMs ? parseInt(envMs, 10) : 600_000;
      timeoutMs = Number.isFinite(parsed) && parsed > 0 ? parsed : 600_000;
    }
    return AbortSignal.timeout(timeoutMs);
  }

  private get headers(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      "X-API-Key": this.apiKey,
    }
  }

  /**
   * Initialize the provider from config or environment variables.
   */
  async initialize(providerConfig: ProviderConfig): Promise<void> {
    this.baseUrl = (providerConfig.baseUrl as string | undefined) || config.cortexBaseUrl
    this.apiKey = providerConfig.apiKey || config.cortexApiKey
    this.ownerId = (providerConfig.ownerId as string | undefined) || config.cortexOwnerId
    this.entityType = config.cortexEntityType
    this.sourceAgent = config.cortexSourceAgent

    // Strip trailing slash for consistent URL construction
    this.baseUrl = this.baseUrl.replace(/\/$/, "")

    if (!this.ownerId) {
      throw new Error("Cortex provider requires CORTEX_OWNER_ID to be set")
    }

    logger.info(`Initialized Cortex provider`, { baseUrl: this.baseUrl, ownerId: this.ownerId })
  }

  /**
   * Ingest sessions into Cortex via the /remember endpoint.
   * Each session is submitted as a full conversation turn list.
   */
  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    const documentIds: string[] = []
    const url = `${this.baseUrl}/api/v1/memories/remember`

    for (const session of sessions) {
      const body = {
        conversation: session.messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
        session_id: `${options.containerTag}:${session.sessionId}`,
        source_session_id: session.sessionId,
        source_type: "batch_ingest",
        owner_id: this.ownerId,
        entity_id: options.containerTag,
        entity_type: this.entityType,
        source_channel: "memorybench",
        source_agent: this.sourceAgent,
        // Speaker identity — enables extraction to use real names instead of
        // generic USER/ASSISTANT role labels (#1886)
        session_date: session.metadata?.formattedDate || session.metadata?.date,
        speaker_a: session.metadata?.speakerA,
        speaker_b: session.metadata?.speakerB,
        metadata: {
          session_date: session.metadata?.date,
          memorybench_container_tag: options.containerTag,
          memorybench_session_id: session.sessionId,
        },
      }

      try {
        const t0 = Date.now()
        const response = await fetch(url, {
          signal: this.timeoutSignal(),
          method: "POST",
          headers: this.headers,
          body: JSON.stringify(body),
          // @ts-ignore — Bun extension: override default 300s fetch timeout
          timeout: 600_000,
        })

        if (!response.ok) {
          const errorText = await response.text()
          logger.error(`Cortex ingest failed for session ${session.sessionId}`, {
            status: response.status,
            error: errorText,
            elapsedMs: Date.now() - t0,
          })
          continue
        }

        const result = await response.json() as Record<string, unknown>
        documentIds.push(session.sessionId)
        logger.info(`Ingested session ${session.sessionId} (${session.messages.length} msgs, ${Date.now() - t0}ms, claims=${result.claims_stored ?? '?'})`)
      } catch (err) {
        logger.error(`Cortex ingest error for session ${session.sessionId}`, {
          error: String(err),
          messages: session.messages.length,
        })
      }
    }

    return { documentIds }
  }

  /**
   * Cortex /remember is synchronous — indexing is immediate.
   * Calls onProgress with all completed IDs right away.
   */
  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    const total = result.documentIds.length
    onProgress?.({ completedIds: [...result.documentIds], failedIds: [], total })
    logger.debug(`Cortex indexing complete (synchronous)`, { total })
  }

  /**
   * Search Cortex memories for a given query.
   */
  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const url = `${this.baseUrl}/api/v1/memories/search`

    const body = {
      query,
      owner_id: this.ownerId,
      // entity_id removed from search — containerTag is a run identifier,
      // not a semantic entity. Sending it filters to a nonexistent entity → 0 results.
      // See: R-454 adapter forensics, golden-v3 0-context root cause.
      limit: options.limit ?? 25,
      include_archived: false,
    }

    try {
      const t0 = Date.now()
      const response = await fetch(url, {
        signal: this.timeoutSignal(30_000),
        method: "POST",
        headers: this.headers,
        body: JSON.stringify(body),
        // @ts-ignore — Bun extension: keep search timeout explicit too
        timeout: 30_000,
      })

      if (!response.ok) {
        const errorText = await response.text()
        logger.error(`Cortex search failed`, { status: response.status, error: errorText, elapsedMs: Date.now() - t0 })
        return []
      }

      const data = (await response.json()) as { items?: unknown[] } | unknown[]
      // Handle both {items: [...]} and [...] response shapes
      const resultCount = Array.isArray(data) ? data.length : (data?.items?.length ?? 0)
      logger.debug(`Search completed (${Date.now() - t0}ms, ${resultCount} results)`)
      if (Array.isArray(data)) return data
      if (data && typeof data === "object" && "items" in data && Array.isArray(data.items)) {
        return data.items
      }
      return []
    } catch (err) {
      logger.error(`Cortex search error`, { error: String(err) })
      return []
    }
  }

  /**
   * Clear all memories for a given containerTag.
   * Paginates via search until no results remain, deleting each memory by ID.
   */
  async clear(containerTag: string): Promise<void> {
    const searchUrl = `${this.baseUrl}/api/v1/memories/search`
    let deleted = 0

    // Pagination loop — keep searching until no results
    while (true) {
      const body = {
        // Use a broad wildcard-like query; empty string is rejected by Cortex (min_length=1).
        // We filter by entity_id server-side to scope deletion to this containerTag only.
        query: "memory",
        owner_id: this.ownerId,
        entity_id: containerTag,
        entity_type: this.entityType,
        limit: 200,
        include_archived: true,
      }

      let items: Array<{ id: string }>
      try {
        const response = await fetch(searchUrl, {
          signal: this.timeoutSignal(),
          method: "POST",
          headers: this.headers,
          body: JSON.stringify(body),
        })

        if (!response.ok) {
          const errorText = await response.text()
          logger.error(`Cortex clear search failed`, { status: response.status, error: errorText })
          break
        }

        const data = (await response.json()) as { items?: Array<{ item_id?: string; id?: string }> } | Array<{ item_id?: string; id?: string }>
        if (Array.isArray(data)) {
          items = data.filter((d: any) => d.item_id || d.id)
        } else if (data && typeof data === "object" && "items" in data && Array.isArray(data.items)) {
          items = data.items.filter((d: any) => d.item_id || d.id)
        } else {
          items = []
        }
      } catch (err) {
        logger.error(`Cortex clear search error`, { error: String(err) })
        break
      }

      if (items.length === 0) break

      // Delete each memory
      for (const item of items) {
        const deleteUrl = `${this.baseUrl}/api/v1/memories/${item.id}?owner_id=${encodeURIComponent(this.ownerId)}`
        try {
          const delResponse = await fetch(deleteUrl, {
          signal: this.timeoutSignal(30_000),
            method: "DELETE",
            headers: this.headers,
          })

          if (!delResponse.ok) {
            const errorText = await delResponse.text()
            logger.error(`Cortex delete failed for memory ${item.id}`, {
              status: delResponse.status,
              error: errorText,
            })
          } else {
            deleted++
          }
        } catch (err) {
          logger.error(`Cortex delete error for memory ${item.id}`, { error: String(err) })
        }
      }

      // If we got fewer than 200, we've seen everything
      if (items.length < 200) break
    }

    logger.info(`Cortex clear complete`, { containerTag, deleted })
  }
}

export default CortexProvider
