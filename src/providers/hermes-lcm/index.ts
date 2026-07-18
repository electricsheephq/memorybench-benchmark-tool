import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process"
import { existsSync } from "node:fs"
import { join } from "node:path"
import { tmpdir } from "node:os"
import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import { logger } from "../../utils/logger"
import { HERMES_LCM_PROMPTS } from "./prompts"

const DEFAULT_REPO = "/Volumes/LEXAR/hermes-work/hermes-lcm"
const INITIALIZE_TIMEOUT_MS = 300_000 // model download/load on first warmup can be slow
const REQUEST_TIMEOUT_MS = 180_000 // voyage is a network provider; give ingest headroom
// Concurrency the provider offers per phase. With one bridge PROCESS per
// container (see below) the phases genuinely parallelize; 3 matches the voyage
// provider concurrency the ingest/search phases run at.
const PROVIDER_CONCURRENCY = 3
// Bounded pool: at most this many live bridge processes. The concurrent executor
// keeps exactly PROVIDER_CONCURRENCY containers in flight, and every request
// bumps its container to most-recently-used, so the two-slot headroom guarantees
// the least-recently-used entry we evict is always a COMPLETED container (never
// one still ingesting) — its on-disk db persists, so a later search re-opens it.
const MAX_BRIDGES = PROVIDER_CONCURRENCY + 2

interface BridgeResponse {
  ok: boolean
  error?: string
  [key: string]: unknown
}

/**
 * One long-lived Python bridge process, dedicated to a single container.
 *
 * Requests are serialized on its own pipe (single in-flight request) and it is
 * crash-loud: if the process exits, the pending call rejects and every
 * subsequent call throws rather than silently degrading.
 */
class BridgeHandle {
  private stdoutBuffer = ""
  private pending: {
    resolve: (r: BridgeResponse) => void
    reject: (e: Error) => void
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
    this.proc.on("error", (err) => {
      this.markDead(new Error(`hermes-lcm bridge (${this.tag}) process error: ${err.message}`))
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
      } catch (e) {
        pending.reject(new Error(`hermes-lcm bridge sent invalid JSON: ${line} (${e})`))
      }
    }
  }

  private markDead(err: Error): void {
    if (!this.deadError) this.deadError = err
    if (this.pending) {
      clearTimeout(this.pending.timer)
      this.pending.reject(err)
      this.pending = null
    }
  }

  /** Serialize requests: one line on the pipe at a time. */
  request(payload: Record<string, unknown>, timeoutMs = REQUEST_TIMEOUT_MS): Promise<BridgeResponse> {
    const run = async (): Promise<BridgeResponse> => {
      if (this.deadError) throw this.deadError
      const resp = await new Promise<BridgeResponse>((resolve, reject) => {
        const timer = setTimeout(
          () =>
            this.markDead(
              new Error(`hermes-lcm bridge (${this.tag}) timed out after ${timeoutMs}ms on ${payload.cmd}`)
            ),
          timeoutMs
        )
        this.pending = { resolve, reject, timer }
        this.proc.stdin.write(JSON.stringify(payload) + "\n")
      })
      if (!resp.ok) {
        throw new Error(`hermes-lcm ${payload.cmd} failed: ${resp.error}`)
      }
      return resp
    }
    // Chain onto the queue so calls never interleave on the shared pipe.
    const result = this.queue.then(run, run)
    this.queue = result.then(
      () => undefined,
      () => undefined
    )
    return result
  }

  /** Reap the process. Best-effort; suppresses the exit-as-crash signal. */
  close(): void {
    this.closed = true
    try {
      this.proc.stdin.end()
    } catch {
      /* ignore */
    }
    try {
      this.proc.kill("SIGTERM")
    } catch {
      /* ignore */
    }
  }
}

/**
 * hermes-lcm memory provider.
 *
 * hermes-lcm is a Python/SQLite lossless-context-management plugin, so the
 * provider drives Python bridges (`bridge/hermes_lcm_bridge.py`) over
 * newline-delimited JSON on stdin/stdout. To let the voyage embedding provider
 * (a network API, ~3 min/question serialized) parallelize, each CONTAINER gets
 * its OWN bridge process — its own db file, its own pipe, no shared state — so
 * up to PROVIDER_CONCURRENCY containers ingest/search concurrently. A bounded
 * LRU pool caps the number of live processes; `provider.clear()` also reaps.
 */
export class HermesLcmProvider implements Provider {
  name = "hermes-lcm"
  prompts = HERMES_LCM_PROMPTS
  concurrency = { default: PROVIDER_CONCURRENCY }

  private python = ""
  private script = ""
  private spawnEnv: Record<string, string> = {}
  // Insertion order == LRU order; a request moves its tag to the end (MRU).
  private handles = new Map<string, BridgeHandle>()

  async initialize(_config: ProviderConfig): Promise<void> {
    const repo = process.env.HERMES_LCM_REPO || DEFAULT_REPO
    this.python =
      process.env.HERMES_LCM_PYTHON || join(repo, ".venv-fastembed", "bin", "python")
    this.script = join(import.meta.dir, "bridge", "hermes_lcm_bridge.py")

    if (!existsSync(this.python)) {
      throw new Error(
        `hermes-lcm python not found at ${this.python}. Set HERMES_LCM_PYTHON or create the fastembed venv (see provider README).`
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

    // Fail fast + disclose the resolved embedder: spawn one probe bridge, warm
    // it, log provider/model/dim, then reap it. Per-container bridges spawn
    // lazily on first ingest/search.
    const probe = this.spawnHandle("__probe__")
    try {
      const resp = await probe.request({ cmd: "initialize" }, INITIALIZE_TIMEOUT_MS)
      logger.info(
        `Initialized hermes-lcm provider (provider=${resp.provider}, model=${resp.model}, dim=${resp.dim}, concurrency=${PROVIDER_CONCURRENCY})`
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

  /** Get (or lazily spawn + initialize) the bridge dedicated to `tag`. */
  private async getHandle(tag: string): Promise<BridgeHandle> {
    const existing = this.handles.get(tag)
    if (existing) {
      if (existing.deadError) throw existing.deadError
      // Move to MRU.
      this.handles.delete(tag)
      this.handles.set(tag, existing)
      return existing
    }
    // Evict LRU (front of the map) until under the cap.
    while (this.handles.size >= MAX_BRIDGES) {
      const lruTag = this.handles.keys().next().value as string | undefined
      if (lruTag === undefined) break
      const lru = this.handles.get(lruTag)!
      this.handles.delete(lruTag)
      lru.close()
    }
    const handle = this.spawnHandle(tag)
    this.handles.set(tag, handle)
    await handle.request({ cmd: "initialize" }, INITIALIZE_TIMEOUT_MS)
    return handle
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    const documentIds: string[] = []
    const handle = await this.getHandle(options.containerTag)
    // The harness calls ingest one session at a time, but honor a batch too.
    for (const session of sessions) {
      const resp = await handle.request({
        cmd: "ingest",
        containerTag: options.containerTag,
        session,
      })
      const ids = (resp.documentIds as string[]) || []
      documentIds.push(...ids)
    }
    return { documentIds }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    // Ingest is fully synchronous (embeddings recorded inline), so indexing is
    // instant. Report every document as completed for the progress tracker.
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const handle = await this.getHandle(options.containerTag)
    const resp = await handle.request({
      cmd: "search",
      containerTag: options.containerTag,
      query,
      limit: options.limit ?? 25,
    })
    if (resp.degraded) {
      logger.debug(`[hermes-lcm] search degraded: ${resp.degraded_reason}`)
    }
    return (resp.results as unknown[]) || []
  }

  async clear(containerTag: string): Promise<void> {
    const handle = this.handles.get(containerTag)
    if (!handle) return
    this.handles.delete(containerTag)
    try {
      if (!handle.deadError) await handle.request({ cmd: "clear", containerTag })
    } catch (e) {
      logger.warn(`Failed to clear hermes-lcm container ${containerTag}: ${e}`)
    } finally {
      handle.close()
    }
  }
}

export default HermesLcmProvider
