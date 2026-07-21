import { execFileSync, spawn } from "node:child_process"
import { createHash } from "node:crypto"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

/**
 * Subscription-CLI transport for benchmark answerer and judge calls.
 *
 * Each call runs in an empty temporary directory, ignores repository/user
 * instructions, disables tools and installed extensions, and receives only an
 * allowlisted environment. Prompts, answers, and raw event payloads are never
 * written to the checkpoint ledger.
 */
export type CliLlmBackend = "codex" | "claude"
export type CliLlmRole = "answerer" | "judge"

export interface CliLlmProvenance {
  transport: "codex-cli" | "claude-cli"
  transportVersion?: string
  model: string
  modelExplicit: boolean
  reasoningEffort?: string
  provider?: string
  serviceTier?: string
  isolated?: boolean
  modelPinSource?: "explicit-cli-argv" | "un-pinned"
  eventUsageCapture?: "codex-jsonl" | "unavailable"
  eventModelField?: "not-emitted-by-codex-jsonl" | "unavailable"
}

export interface CliProviderUsage {
  inputTokens: number
  cachedInputTokens: number
  outputTokens: number
  reasoningOutputTokens: number
}

export interface CodexJsonlTelemetry {
  threadId?: string
  usage?: CliProviderUsage
  eventCount: number
  errorEventCount: number
  eventStreamSha256: string
}

export interface CliAttemptTelemetry extends CodexJsonlTelemetry {
  attempt: number
  status: "completed" | "failed" | "timed_out" | "spawn_error"
  startedAt: string
  durationMs: number
  errorCode?: "process_exit" | "timeout" | "spawn_error" | "output_read"
}

export interface CliCallTelemetry {
  version: "memorybench-cli-call-v1"
  role: CliLlmRole
  transport: "codex-cli" | "claude-cli"
  transportVersion?: string
  requested: {
    model: string
    modelExplicit: boolean
    reasoningEffort: string
    provider: string
    serviceTier: string
    pinSource: "explicit-cli-argv" | "un-pinned"
  }
  eventModelField: "not-emitted-by-codex-jsonl" | "unavailable"
  attempts: CliAttemptTelemetry[]
  usage: CliProviderUsage
  usageComplete: boolean
  totalDurationMs: number
  retryCount: number
}

export class CliCallError extends Error {
  constructor(
    message: string,
    readonly telemetry: CliCallTelemetry
  ) {
    super(message)
    this.name = "CliCallError"
  }
}

export function cliCallTelemetryFromError(error: unknown): CliCallTelemetry | undefined {
  return error instanceof CliCallError ? error.telemetry : undefined
}

export interface CliLedgerPhase {
  status: string
  llmCall?: CliCallTelemetry
  llmCalls?: CliCallTelemetry[]
}

export function cliCallsFromPhase(phase: CliLedgerPhase): CliCallTelemetry[] {
  return phase.llmCalls?.length ? phase.llmCalls : phase.llmCall ? [phase.llmCall] : []
}

export function summarizeCliLedger(phases: CliLedgerPhase[]): {
  calls: CliCallTelemetry[]
  callCount: number
  retryCount: number
  executionIdentityCount: number
  mixedExecutionIdentity: boolean
  callLedgerComplete: boolean
} {
  const calls = phases.flatMap(cliCallsFromPhase)
  const completed = phases.filter((phase) => phase.status === "completed")
  const executionIdentityCount = new Set(
    calls.map((call) =>
      JSON.stringify([
        call.transport,
        call.transportVersion,
        call.requested.model,
        call.requested.modelExplicit,
        call.requested.reasoningEffort,
        call.requested.provider,
        call.requested.serviceTier,
        call.requested.pinSource,
      ])
    )
  ).size
  return {
    calls,
    callCount: calls.length,
    retryCount: calls.reduce((sum, call) => sum + call.retryCount, 0),
    executionIdentityCount,
    mixedExecutionIdentity: executionIdentityCount > 1,
    callLedgerComplete:
      completed.length > 0 &&
      completed.every((phase) => cliCallsFromPhase(phase).at(-1)?.usageComplete === true) &&
      calls.every((call) => call.usageComplete),
  }
}

type ReconciledProvenance = {
  transport: "ai-sdk" | "codex-cli" | "claude-cli" | "mixed"
  transportVersion?: string
  model: string
  modelExplicit: boolean
  reasoningEffort?: string
  provider?: string
  serviceTier?: string
  isolated?: boolean
  modelPinSource?: "explicit-cli-argv" | "un-pinned" | "mixed"
  eventUsageCapture?: "codex-jsonl" | "unavailable"
  eventModelField?: "not-emitted-by-codex-jsonl" | "unavailable"
}

/** Replace configuration-time labels with the identities stored by real calls. */
export function reconcileCliProvenanceIdentity(
  provenance: ReconciledProvenance,
  calls: CliCallTelemetry[]
): void {
  if (calls.length === 0) return
  const first = calls[0]!
  const identityKeys = new Set(
    calls.map((call) => JSON.stringify([call.transport, call.transportVersion, call.requested]))
  )

  if (identityKeys.size === 1) {
    provenance.transport = first.transport
    provenance.transportVersion = first.transportVersion
    provenance.model = first.requested.model
    provenance.modelExplicit = first.requested.modelExplicit
    provenance.reasoningEffort = first.requested.reasoningEffort
    provenance.provider = first.requested.provider
    provenance.serviceTier = first.requested.serviceTier
    provenance.isolated = true
    provenance.modelPinSource = first.requested.pinSource
    provenance.eventUsageCapture = first.transport === "codex-cli" ? "codex-jsonl" : "unavailable"
    provenance.eventModelField = first.eventModelField
    return
  }

  const oneOrMixed = (values: Array<string | undefined>): string | undefined => {
    const unique = [...new Set(values.filter((value): value is string => Boolean(value)))]
    return unique.length === 1 ? unique[0] : unique.length > 1 ? "mixed" : undefined
  }
  const transports = new Set(calls.map((call) => call.transport))
  const pinSources = new Set(calls.map((call) => call.requested.pinSource))
  provenance.transport = transports.size === 1 ? first.transport : "mixed"
  provenance.transportVersion = oneOrMixed(calls.map((call) => call.transportVersion))
  provenance.model = "mixed stored CLI call identities"
  provenance.modelExplicit = calls.every((call) => call.requested.modelExplicit)
  provenance.reasoningEffort = oneOrMixed(calls.map((call) => call.requested.reasoningEffort))
  provenance.provider = oneOrMixed(calls.map((call) => call.requested.provider))
  provenance.serviceTier = oneOrMixed(calls.map((call) => call.requested.serviceTier))
  provenance.isolated = true
  provenance.modelPinSource = pinSources.size === 1 ? first.requested.pinSource : "mixed"
  provenance.eventUsageCapture = calls.every((call) => call.transport === "codex-cli")
    ? "codex-jsonl"
    : "unavailable"
  provenance.eventModelField = calls.every(
    (call) => call.eventModelField === "not-emitted-by-codex-jsonl"
  )
    ? "not-emitted-by-codex-jsonl"
    : "unavailable"
}

const ALLOWED_CHILD_ENV = [
  "PATH",
  "HOME",
  "USER",
  "LOGNAME",
  "SHELL",
  "TMPDIR",
  "TMP",
  "TEMP",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "TERM",
  "COLORTERM",
  "NO_COLOR",
  "CODEX_HOME",
  "CLAUDE_CONFIG_DIR",
  "XDG_CONFIG_HOME",
  "XDG_CACHE_HOME",
  "SSL_CERT_FILE",
  "SSL_CERT_DIR",
] as const

/**
 * Keep only executable/auth-location and locale variables. Provider keys,
 * benchmark configuration, database URLs, and arbitrary host secrets do not
 * cross the subprocess boundary.
 */
export function buildCliEnvironment(
  source: NodeJS.ProcessEnv | Record<string, string | undefined> = process.env
): NodeJS.ProcessEnv {
  const result: NodeJS.ProcessEnv = {}
  for (const name of ALLOWED_CHILD_ENV) {
    const value = source[name]
    if (value !== undefined) result[name] = value
  }
  return result
}

const CLI_VERSION_CACHE = new Map<string, string | undefined>()

function cliTransportVersion(backend: CliLlmBackend): string | undefined {
  const cacheKey = [
    backend,
    process.env.PATH,
    process.env.HOME,
    process.env.CODEX_HOME,
    process.env.CLAUDE_CONFIG_DIR,
  ].join("\0")
  if (CLI_VERSION_CACHE.has(cacheKey)) return CLI_VERSION_CACHE.get(cacheKey)
  try {
    const version = execFileSync(backend, ["--version"], {
      encoding: "utf8",
      env: buildCliEnvironment(),
      timeout: 10_000,
      stdio: ["ignore", "pipe", "ignore"],
    }).trim()
    CLI_VERSION_CACHE.set(cacheKey, version || undefined)
  } catch {
    CLI_VERSION_CACHE.set(cacheKey, undefined)
  }
  return CLI_VERSION_CACHE.get(cacheKey)
}

export function cliLlmBackend(): CliLlmBackend | null {
  const backend = (process.env.HERMES_MB_LLM_CLI || "").trim().toLowerCase()
  return backend === "codex" || backend === "claude" ? backend : null
}

function modelForRole(backend: CliLlmBackend, role: CliLlmRole): string | undefined {
  if (backend === "codex") {
    if (role === "judge") {
      return (
        process.env.HERMES_MB_CODEX_JUDGE_MODEL?.trim() ||
        process.env.HERMES_MB_CODEX_MODEL?.trim() ||
        undefined
      )
    }
    return process.env.HERMES_MB_CODEX_MODEL?.trim() || undefined
  }
  if (role === "judge") {
    return (
      process.env.HERMES_MB_CLAUDE_JUDGE_MODEL?.trim() ||
      process.env.HERMES_MB_CLAUDE_MODEL?.trim() ||
      undefined
    )
  }
  return process.env.HERMES_MB_CLAUDE_MODEL?.trim() || undefined
}

function effortForRole(role: CliLlmRole): string {
  if (role === "judge") {
    return process.env.HERMES_MB_CODEX_JUDGE_EFFORT || process.env.HERMES_MB_CODEX_EFFORT || "low"
  }
  return process.env.HERMES_MB_CODEX_ANSWER_EFFORT || "medium"
}

function requestedExecution(
  backend: CliLlmBackend,
  role: CliLlmRole,
  modelOverride?: string,
  effortOverride?: string
): CliCallTelemetry["requested"] {
  const model = modelOverride?.trim() || modelForRole(backend, role)
  if (backend === "claude") {
    return {
      model: model || "claude default (un-pinned)",
      modelExplicit: Boolean(model),
      reasoningEffort: "unavailable",
      provider: "anthropic",
      serviceTier: "unavailable",
      pinSource: model ? "explicit-cli-argv" : "un-pinned",
    }
  }
  return {
    model: model || "codex default (un-pinned)",
    modelExplicit: Boolean(model),
    reasoningEffort: effortOverride || effortForRole(role),
    provider: process.env.HERMES_MB_CODEX_PROVIDER || "openai",
    serviceTier: process.env.HERMES_MB_CODEX_SERVICE_TIER || "priority",
    pinSource: model ? "explicit-cli-argv" : "un-pinned",
  }
}

/** Human-readable requested identity for progress output. */
export function cliLlmModelId(role: CliLlmRole = "answerer"): string {
  const backend = cliLlmBackend()
  if (!backend) return "n/a"
  const requested = requestedExecution(backend, role)
  return `${requested.model} (via ${backend === "codex" ? "codex exec" : "claude -p"})`
}

export function cliLlmProvenance(
  role: CliLlmRole,
  modelOverride?: string
): CliLlmProvenance | null {
  const backend = cliLlmBackend()
  if (!backend) return null
  const requested = requestedExecution(backend, role, modelOverride)
  return {
    transport: backend === "codex" ? "codex-cli" : "claude-cli",
    transportVersion: cliTransportVersion(backend),
    model: requested.model,
    modelExplicit: requested.modelExplicit,
    reasoningEffort:
      requested.reasoningEffort === "unavailable" ? undefined : requested.reasoningEffort,
    provider: requested.provider,
    serviceTier: requested.serviceTier,
    isolated: true,
    modelPinSource: requested.pinSource,
    eventUsageCapture: backend === "codex" ? "codex-jsonl" : "unavailable",
    eventModelField: backend === "codex" ? "not-emitted-by-codex-jsonl" : "unavailable",
  }
}

export interface CliCompleteOptions {
  effort?: string
  model?: string
  timeoutMs?: number
  retry?: boolean
  role?: CliLlmRole
  onTelemetry?: (telemetry: CliCallTelemetry) => void
}

function tomlString(value: string): string {
  return JSON.stringify(value)
}

export function buildCodexCompletionArgs(
  outFile: string,
  effort: string,
  isolatedCwd: string,
  model?: string,
  provider = process.env.HERMES_MB_CODEX_PROVIDER || "openai",
  serviceTier = process.env.HERMES_MB_CODEX_SERVICE_TIER || "priority"
): string[] {
  const args = [
    "exec",
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "-s",
    "read-only",
    "--ephemeral",
    "--json",
    "-c",
    `model_reasoning_effort=${effort}`,
    "-c",
    `model_provider=${tomlString(provider)}`,
    "-c",
    `service_tier=${tomlString(serviceTier)}`,
    "-c",
    'approval_policy="never"',
    "-c",
    'web_search="disabled"',
    "-c",
    "features.shell_tool=false",
    "-c",
    "features.multi_agent=false",
    "-c",
    "features.multi_agent_v2=false",
    "-c",
    "features.apps=false",
    "-c",
    "features.plugins=false",
    "-c",
    "features.remote_plugin=false",
    "-c",
    "features.plugin_sharing=false",
    "-c",
    "features.hooks=false",
    "-c",
    "features.skill_mcp_dependency_install=false",
    "-c",
    "features.code_mode=false",
    "-c",
    "features.code_mode_only=false",
    "-c",
    "features.tool_search=false",
    "-c",
    "features.standalone_web_search=false",
    "-c",
    "skills.include_instructions=false",
    "-c",
    "include_apps_instructions=false",
    "-c",
    "include_environment_context=false",
    "-c",
    "include_collaboration_mode_instructions=false",
    "-C",
    isolatedCwd,
    "-o",
    outFile,
  ]
  if (model) args.push("-m", model)
  args.push("-")
  return args
}

export function buildClaudeCompletionArgs(mcpConfig: string, model?: string): string[] {
  const args = [
    "-p",
    "--output-format",
    "text",
    "--permission-mode",
    "dontAsk",
    "--strict-mcp-config",
    "--mcp-config",
    mcpConfig,
    "--tools",
    "",
  ]
  if (model) args.push("--model", model)
  return args
}

/** Parse a Codex event stream into bounded metadata; raw events are discarded. */
export function parseCodexJsonlTelemetry(jsonl: string): CodexJsonlTelemetry {
  let threadId: string | undefined
  let usage: CliProviderUsage | undefined
  let eventCount = 0
  let errorEventCount = 0
  for (const line of jsonl.split(/\r?\n/)) {
    if (!line.trim()) continue
    eventCount++
    try {
      const event = JSON.parse(line) as Record<string, unknown>
      if (event.type === "thread.started" && typeof event.thread_id === "string") {
        threadId = event.thread_id
      }
      if (event.type === "error") errorEventCount++
      if (event.type === "turn.completed" && event.usage && typeof event.usage === "object") {
        const raw = event.usage as Record<string, unknown>
        const counter = (value: unknown): number | undefined =>
          typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined
        const inputTokens = counter(raw.input_tokens)
        const cachedInputTokens = counter(raw.cached_input_tokens)
        const outputTokens = counter(raw.output_tokens)
        const reasoningOutputTokens = counter(raw.reasoning_output_tokens)
        if (
          inputTokens !== undefined &&
          cachedInputTokens !== undefined &&
          outputTokens !== undefined &&
          reasoningOutputTokens !== undefined
        ) {
          usage = { inputTokens, cachedInputTokens, outputTokens, reasoningOutputTokens }
        }
      }
    } catch {
      // Counts and the digest make malformed output visible without retaining it.
    }
  }
  return {
    threadId,
    usage,
    eventCount,
    errorEventCount,
    eventStreamSha256: createHash("sha256").update(jsonl).digest("hex"),
  }
}

type ProcessFailureKind = "process_exit" | "timeout" | "spawn_error" | "output_read"

class ProcessFailure extends Error {
  constructor(
    message: string,
    readonly kind: ProcessFailureKind
  ) {
    super(message)
    this.name = "ProcessFailure"
  }
}

function resolvedTimeout(override?: number): number {
  const value = override ?? Number(process.env.HERMES_MB_CLI_TIMEOUT_MS || 180_000)
  return Number.isFinite(value) && value > 0 ? value : 180_000
}

function runProcess<T>(
  command: string,
  args: string[],
  prompt: string,
  options: {
    env: NodeJS.ProcessEnv
    cwd?: string
    timeoutMs?: number
    readResult: () => T
    onStdout?: (chunk: string) => void
  }
): Promise<T> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      env: options.env,
      cwd: options.cwd,
      stdio: ["pipe", "pipe", "pipe"],
    })
    let settled = false
    const finish = (error?: Error, result?: T): void => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      if (error) reject(error)
      else resolve(result as T)
    }
    const timeoutMs = resolvedTimeout(options.timeoutMs)
    const timer = setTimeout(() => {
      child.kill("SIGKILL")
      finish(new ProcessFailure(`${command} timed out after ${timeoutMs}ms`, "timeout"))
    }, timeoutMs)

    child.stdout.on("data", (data) => options.onStdout?.(data.toString()))
    // Drain stderr without persisting it. CLI errors can contain prompt or host data.
    child.stderr.on("data", () => {})
    child.stdin.on("error", () => {})
    child.on("error", (error) => {
      finish(new ProcessFailure(`${command} spawn error: ${error.message}`, "spawn_error"))
    })
    child.on("close", (code) => {
      if (settled) return
      if (code !== 0) {
        finish(new ProcessFailure(`${command} exited ${code}`, "process_exit"))
        return
      }
      try {
        const result = options.readResult()
        if (typeof result === "string" && !result.trim()) {
          finish(new ProcessFailure(`${command} produced empty output`, "output_read"))
          return
        }
        finish(undefined, result)
      } catch (error) {
        finish(
          new ProcessFailure(
            `${command} output read failed: ${error instanceof Error ? error.message : String(error)}`,
            "output_read"
          )
        )
      }
    })
    child.stdin.end(prompt)
  })
}

class CliAttemptError extends Error {
  constructor(
    message: string,
    readonly telemetry: CliAttemptTelemetry
  ) {
    super(message)
    this.name = "CliAttemptError"
  }
}

function attemptErrorCode(error: unknown): CliAttemptTelemetry["errorCode"] {
  if (error instanceof ProcessFailure) return error.kind
  return "process_exit"
}

function failedStatus(error: unknown): CliAttemptTelemetry["status"] {
  const code = attemptErrorCode(error)
  return code === "timeout" ? "timed_out" : code === "spawn_error" ? "spawn_error" : "failed"
}

async function codexAttempt(
  prompt: string,
  requested: CliCallTelemetry["requested"],
  attempt: number,
  timeoutMs?: number
): Promise<{ text: string; telemetry: CliAttemptTelemetry }> {
  const dir = mkdtempSync(join(tmpdir(), "memorybench-codex-"))
  const outFile = join(dir, "out.txt")
  const args = buildCodexCompletionArgs(
    outFile,
    requested.reasoningEffort,
    dir,
    requested.modelExplicit ? requested.model : undefined,
    requested.provider,
    requested.serviceTier
  )
  const startedAt = new Date().toISOString()
  const started = Date.now()
  let stdout = ""
  const telemetry = (
    status: CliAttemptTelemetry["status"],
    errorCode?: CliAttemptTelemetry["errorCode"]
  ): CliAttemptTelemetry => ({
    attempt,
    status,
    startedAt,
    durationMs: Date.now() - started,
    ...parseCodexJsonlTelemetry(stdout),
    errorCode,
  })

  try {
    const text = await runProcess("codex", args, prompt, {
      env: buildCliEnvironment(),
      cwd: dir,
      timeoutMs,
      readResult: () => readFileSync(outFile, "utf8"),
      onStdout: (chunk) => {
        stdout += chunk
      },
    })
    return { text, telemetry: telemetry("completed") }
  } catch (error) {
    throw new CliAttemptError(
      error instanceof Error ? error.message : String(error),
      telemetry(failedStatus(error), attemptErrorCode(error))
    )
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

function loadClaudeOauthToken(): string | undefined {
  const account = process.env.USER || process.env.LOGNAME
  if (!account) return undefined
  try {
    const raw = execFileSync(
      "security",
      ["find-generic-password", "-s", "Claude Code-credentials", "-a", account, "-w"],
      {
        encoding: "utf8",
        env: buildCliEnvironment(),
        timeout: 10_000,
        stdio: ["ignore", "pipe", "ignore"],
      }
    )
    const parsed = JSON.parse(raw) as { claudeAiOauth?: { accessToken?: unknown } }
    const token = parsed.claudeAiOauth?.accessToken
    return typeof token === "string" && token ? token : undefined
  } catch {
    return undefined
  }
}

async function claudeAttempt(
  prompt: string,
  requested: CliCallTelemetry["requested"],
  attempt: number,
  timeoutMs?: number
): Promise<{ text: string; telemetry: CliAttemptTelemetry }> {
  const dir = mkdtempSync(join(tmpdir(), "memorybench-claude-"))
  const mcpConfig = join(dir, "mcp.json")
  writeFileSync(join(dir, "settings.json"), "{}\n")
  writeFileSync(mcpConfig, '{"mcpServers":{}}\n')
  const args = buildClaudeCompletionArgs(
    mcpConfig,
    requested.modelExplicit ? requested.model : undefined
  )
  const env = buildCliEnvironment()
  env.CLAUDE_CONFIG_DIR = dir
  const oauthToken = loadClaudeOauthToken()
  if (oauthToken) env.CLAUDE_CODE_OAUTH_TOKEN = oauthToken
  const startedAt = new Date().toISOString()
  const started = Date.now()
  const emptyTelemetry = (): CodexJsonlTelemetry => ({
    eventCount: 0,
    errorEventCount: 0,
    eventStreamSha256: createHash("sha256").update("").digest("hex"),
  })
  const telemetry = (
    status: CliAttemptTelemetry["status"],
    errorCode?: CliAttemptTelemetry["errorCode"]
  ): CliAttemptTelemetry => ({
    attempt,
    status,
    startedAt,
    durationMs: Date.now() - started,
    ...emptyTelemetry(),
    errorCode,
  })
  let stdout = ""

  try {
    const text = await runProcess("claude", args, prompt, {
      env,
      cwd: dir,
      timeoutMs,
      readResult: () => stdout,
      onStdout: (chunk) => {
        stdout += chunk
      },
    })
    return { text, telemetry: telemetry("completed") }
  } catch (error) {
    throw new CliAttemptError(
      error instanceof Error ? error.message : String(error),
      telemetry(failedStatus(error), attemptErrorCode(error))
    )
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

function aggregateCliTelemetry(
  backend: CliLlmBackend,
  role: CliLlmRole,
  requested: CliCallTelemetry["requested"],
  attempts: CliAttemptTelemetry[]
): CliCallTelemetry {
  const usage = attempts.reduce<CliProviderUsage>(
    (total, attempt) => ({
      inputTokens: total.inputTokens + (attempt.usage?.inputTokens || 0),
      cachedInputTokens: total.cachedInputTokens + (attempt.usage?.cachedInputTokens || 0),
      outputTokens: total.outputTokens + (attempt.usage?.outputTokens || 0),
      reasoningOutputTokens:
        total.reasoningOutputTokens + (attempt.usage?.reasoningOutputTokens || 0),
    }),
    { inputTokens: 0, cachedInputTokens: 0, outputTokens: 0, reasoningOutputTokens: 0 }
  )
  return {
    version: "memorybench-cli-call-v1",
    role,
    transport: backend === "codex" ? "codex-cli" : "claude-cli",
    transportVersion: cliTransportVersion(backend),
    requested,
    eventModelField: backend === "codex" ? "not-emitted-by-codex-jsonl" : "unavailable",
    attempts,
    usage,
    usageComplete:
      backend === "codex" &&
      attempts.length > 0 &&
      attempts.every((attempt) => attempt.status === "completed" && Boolean(attempt.usage)),
    totalDurationMs: attempts.reduce((sum, attempt) => sum + attempt.durationMs, 0),
    retryCount: Math.max(0, attempts.length - 1),
  }
}

function emitCliTelemetry(
  callback: CliCompleteOptions["onTelemetry"],
  telemetry: CliCallTelemetry
): void {
  if (!callback) return
  try {
    callback(telemetry)
  } catch {
    // Observability must not change benchmark output.
  }
}

export async function cliComplete(
  prompt: string,
  options: CliCompleteOptions = {}
): Promise<string> {
  const backend = cliLlmBackend()
  if (!backend) throw new Error("cliComplete called but HERMES_MB_LLM_CLI is not codex|claude")
  const role = options.role || "answerer"
  const requested = requestedExecution(backend, role, options.model, options.effort)
  const attempts: CliAttemptTelemetry[] = []
  const maxAttempts = options.retry === false ? 1 : 2
  let lastError: unknown

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const result =
        backend === "codex"
          ? await codexAttempt(prompt, requested, attempt, options.timeoutMs)
          : await claudeAttempt(prompt, requested, attempt, options.timeoutMs)
      attempts.push(result.telemetry)
      const telemetry = aggregateCliTelemetry(backend, role, requested, attempts)
      emitCliTelemetry(options.onTelemetry, telemetry)
      return result.text.trim()
    } catch (error) {
      lastError = error
      if (error instanceof CliAttemptError) attempts.push(error.telemetry)
      if (attempt < maxAttempts) await new Promise((resolve) => setTimeout(resolve, 1_000))
    }
  }

  const telemetry = aggregateCliTelemetry(backend, role, requested, attempts)
  emitCliTelemetry(options.onTelemetry, telemetry)
  throw new CliCallError(
    lastError instanceof Error ? lastError.message : String(lastError),
    telemetry
  )
}
