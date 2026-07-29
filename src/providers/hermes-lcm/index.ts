import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process"
import { existsSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type {
  IndexingProgressCallback,
  IngestOptions,
  IngestResult,
  Provider,
  ProviderConfig,
  SearchOptions,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import { logger } from "../../utils/logger"
import { HERMES_LCM_PROMPTS } from "./prompts"

const DEFAULT_REPO = "/Volumes/LEXAR/hermes-work/hermes-lcm"
const INITIALIZE_TIMEOUT_MS = 300_000
const REQUEST_TIMEOUT_MS = 180_000
const PROVIDER_CONCURRENCY = 3
const MAX_BRIDGES = PROVIDER_CONCURRENCY + 2

interface BridgeResponse {
  ok: boolean
  error?: string
  [key: string]: unknown
}

/**
 * Return exactly the ordinary result array that the historical M450 search
 * phase persisted after normalizing the bridge response. Provider provenance
 * deliberately stays outside scored context and prompt bytes.
 */
export function normalizeHermesSearchResponse(response: BridgeResponse): unknown[] {
  if (!Array.isArray(response.results)) {
    throw new Error("hermes-lcm search response did not contain a results array")
  }
  return response.results
}

/** One long-lived JSONL bridge process dedicated to one database container. */
class BridgeHandle {
  private stdoutBuffer = ""
  private pending: {
    resolve: (response: BridgeResponse) => void
    reject: (error: Error) => void
    timer: ReturnType<typeof setTimeout>
  } | null = null
  private queue: Promise<unknown> = Promise.resolve()
  deadError: Error | null = null
  private closed = false

  constructor(
    private readonly proc: ChildProcessWithoutNullStreams,
    private readonly tag: string
  ) {
    this.proc.stdout.setEncoding("utf8")
    this.proc.stderr.setEncoding("utf8")
    this.proc.stdout.on("data", (chunk: string) => this.onStdout(chunk))
    this.proc.stderr.on("data", (chunk: string) => {
      for (const line of chunk.split("\n")) {
        if (line.trim()) logger.debug(`[hermes-lcm:${this.tag}] ${line}`)
      }
    })
    this.proc.on("exit", (code, signal) => {
      if (this.closed) return
      this.markDead(
        new Error(`hermes-lcm bridge (${this.tag}) exited (code=${code}, signal=${signal})`)
      )
    })
    this.proc.on("error", (error) => {
      this.markDead(new Error(`hermes-lcm bridge (${this.tag}) process error: ${error.message}`))
    })
  }

  private onStdout(chunk: string): void {
    this.stdoutBuffer += chunk
    let newlineIndex: number
    while ((newlineIndex = this.stdoutBuffer.indexOf("\n")) !== -1) {
      const line = this.stdoutBuffer.slice(0, newlineIndex).trim()
      this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1)
      if (!line) continue

      const pending = this.pending
      this.pending = null
      if (!pending) {
        logger.warn(`[hermes-lcm:${this.tag}] unexpected bridge output: ${line}`)
        continue
      }
      clearTimeout(pending.timer)
      try {
        pending.resolve(JSON.parse(line) as BridgeResponse)
      } catch (error) {
        pending.reject(new Error(`hermes-lcm bridge sent invalid JSON: ${line} (${error})`))
      }
    }
  }

  private markDead(error: Error): void {
    if (!this.deadError) this.deadError = error
    if (this.pending) {
      clearTimeout(this.pending.timer)
      this.pending.reject(error)
      this.pending = null
    }
  }

  request(
    payload: Record<string, unknown>,
    timeoutMs = REQUEST_TIMEOUT_MS
  ): Promise<BridgeResponse> {
    const run = async (): Promise<BridgeResponse> => {
      if (this.deadError) throw this.deadError
      const response = await new Promise<BridgeResponse>((resolve, reject) => {
        const timer = setTimeout(
          () =>
            this.markDead(
              new Error(
                `hermes-lcm bridge (${this.tag}) timed out after ${timeoutMs}ms on ${payload.cmd}`
              )
            ),
          timeoutMs
        )
        this.pending = { resolve, reject, timer }
        this.proc.stdin.write(`${JSON.stringify(payload)}\n`)
      })
      if (!response.ok) {
        throw new Error(`hermes-lcm ${payload.cmd} failed: ${response.error}`)
      }
      return response
    }

    const result = this.queue.then(run, run)
    this.queue = result.then(
      () => undefined,
      () => undefined
    )
    return result
  }

  close(): void {
    this.closed = true
    try {
      this.proc.stdin.end()
    } catch {
      // Best-effort process cleanup.
    }
    try {
      this.proc.kill("SIGTERM")
    } catch {
      // Best-effort process cleanup.
    }
  }
}

/** Minimal ordinary-path adapter for the current MemoryBench Provider contract. */
export class HermesLcmProvider implements Provider {
  name = "hermes-lcm"
  prompts = HERMES_LCM_PROMPTS
  concurrency = { default: PROVIDER_CONCURRENCY }

  private python = ""
  private script = ""
  private spawnEnv: Record<string, string> = {}
  private handles = new Map<string, BridgeHandle>()

  async initialize(_config: ProviderConfig): Promise<void> {
    const repo = process.env.HERMES_LCM_REPO || DEFAULT_REPO
    this.python = process.env.HERMES_LCM_PYTHON || join(repo, ".venv-fastembed", "bin", "python")
    this.script = join(import.meta.dir, "bridge", "hermes_lcm_bridge.py")

    if (!existsSync(this.python)) {
      throw new Error(
        `hermes-lcm python not found at ${this.python}. Set HERMES_LCM_PYTHON explicitly.`
      )
    }
    if (!existsSync(this.script)) {
      throw new Error(`hermes-lcm bridge script not found at ${this.script}`)
    }

    const workdir = process.env.HERMES_MB_WORKDIR || join(tmpdir(), "hermes-lcm-mb")
    this.spawnEnv = {
      ...process.env,
      HERMES_LCM_REPO: repo,
      HERMES_MB_WORKDIR: workdir,
      HERMES_MB_PROVIDER: process.env.HERMES_MB_PROVIDER || "fastembed",
      PYTHONUNBUFFERED: "1",
    } as Record<string, string>

    const probe = this.spawnHandle("__probe__")
    try {
      const response = await probe.request({ cmd: "initialize" }, INITIALIZE_TIMEOUT_MS)
      logger.info(
        `Initialized hermes-lcm provider (provider=${response.provider}, model=${response.model}, dim=${response.dim}, concurrency=${PROVIDER_CONCURRENCY})`
      )
    } finally {
      probe.close()
    }
  }

  private spawnHandle(tag: string): BridgeHandle {
    const proc = spawn(this.python, [this.script, "serve"], {
      env: this.spawnEnv,
    }) as ChildProcessWithoutNullStreams
    return new BridgeHandle(proc, tag)
  }

  private async getHandle(tag: string): Promise<BridgeHandle> {
    const existing = this.handles.get(tag)
    if (existing) {
      if (existing.deadError) throw existing.deadError
      this.handles.delete(tag)
      this.handles.set(tag, existing)
      return existing
    }

    while (this.handles.size >= MAX_BRIDGES) {
      const leastRecentTag = this.handles.keys().next().value as string | undefined
      if (leastRecentTag === undefined) break
      const leastRecent = this.handles.get(leastRecentTag)!
      this.handles.delete(leastRecentTag)
      leastRecent.close()
    }

    const handle = this.spawnHandle(tag)
    this.handles.set(tag, handle)
    await handle.request({ cmd: "initialize" }, INITIALIZE_TIMEOUT_MS)
    return handle
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    const documentIds: string[] = []
    const handle = await this.getHandle(options.containerTag)
    for (const session of sessions) {
      const response = await handle.request({
        cmd: "ingest",
        containerTag: options.containerTag,
        session,
      })
      documentIds.push(...((response.documentIds as string[]) || []))
    }
    return { documentIds }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const handle = await this.getHandle(options.containerTag)
    const response = await handle.request({
      cmd: "search",
      containerTag: options.containerTag,
      query,
      limit: options.limit ?? 25,
    })
    if (response.degraded) {
      logger.debug(`[hermes-lcm] search degraded: ${response.degraded_reason}`)
    }
    return normalizeHermesSearchResponse(response)
  }

  async clear(containerTag: string): Promise<void> {
    const handle = this.handles.get(containerTag)
    if (!handle) return
    this.handles.delete(containerTag)
    try {
      if (!handle.deadError) await handle.request({ cmd: "clear", containerTag })
    } catch (error) {
      logger.warn(`Failed to clear hermes-lcm container ${containerTag}: ${error}`)
    } finally {
      handle.close()
    }
  }

  /** Close bridge processes without deleting their on-disk stores. */
  close(): void {
    for (const handle of this.handles.values()) {
      handle.close()
    }
    this.handles.clear()
  }
}

export default HermesLcmProvider
