import { afterEach, describe, expect, test } from "bun:test"
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { tmpdir } from "node:os"
import {
  buildCliEnvironment,
  buildClaudeCompletionArgs,
  buildCodexCompletionArgs,
  cliComplete,
  CliCallError,
  parseCodexJsonlTelemetry,
  summarizeCliLedger,
  type CliCallTelemetry,
} from "./cli-llm"

const originalEnv = { ...process.env }
const tempPaths: string[] = []

function installFakeCodex(events: unknown[] = []): void {
  const dir = mkdtempSync(join(tmpdir(), "memorybench-fake-codex-"))
  tempPaths.push(dir)
  const executable = join(dir, "codex")
  writeFileSync(
    executable,
    `#!/bin/sh
if [ "$1" = "--version" ]; then
  printf 'codex-cli 1.2.3\\n'
  exit 0
fi
if [ -n "\${OPENAI_API_KEY+x}" ] || [ -n "\${VOYAGE_API_KEY+x}" ]; then
  exit 91
fi
out=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    shift
    out="$1"
  fi
  shift
done
IFS= read -r _prompt || true
printf 'answer from fake CLI' > "$out"
printf '%s\\n' '{"type":"thread.started","thread_id":"thread-1"}'
printf '%s\\n' '{"type":"item.completed","item":{"type":"agent_message","text":"answer from fake CLI"}}'
printf '%s\\n' '{"type":"turn.completed","usage":{"input_tokens":12,"cached_input_tokens":3,"output_tokens":4,"reasoning_output_tokens":2}}'
${events.map((event) => `printf '%s\\n' '${JSON.stringify(event).replaceAll("'", "'\\''")}'`).join("\n")}
`,
    { mode: 0o755 }
  )
  chmodSync(executable, 0o755)
  process.env.PATH = `${dir}:${originalEnv.PATH || ""}`
  process.env.HERMES_MB_LLM_CLI = "codex"
}

afterEach(() => {
  for (const key of Object.keys(process.env)) delete process.env[key]
  Object.assign(process.env, originalEnv)
  for (const path of tempPaths.splice(0)) rmSync(path, { recursive: true, force: true })
})

describe("isolated CLI completion", () => {
  test("pins the requested execution and disables tools, apps, plugins, hooks, and skills", () => {
    const args = buildCodexCompletionArgs(
      "/tmp/out.txt",
      "low",
      "/tmp/isolated",
      "gpt-test",
      "openai",
      "priority"
    )
    expect(args).toContain("--ignore-user-config")
    expect(args).toContain("--ignore-rules")
    expect(args).toContain("--ephemeral")
    expect(args).toContain("--json")
    expect(args).toContain("model_reasoning_effort=low")
    expect(args).toContain('model_provider="openai"')
    expect(args).toContain('service_tier="priority"')
    for (const flag of [
      "features.shell_tool=false",
      "features.apps=false",
      "features.plugins=false",
      "features.remote_plugin=false",
      "features.plugin_sharing=false",
      "features.hooks=false",
      "features.skill_mcp_dependency_install=false",
      "skills.include_instructions=false",
      "include_environment_context=false",
    ]) {
      expect(args).toContain(flag)
    }
    expect(args.slice(-3)).toEqual(["-m", "gpt-test", "-"])
  })

  test("isolates Claude settings, MCP servers, tools, and model selection", () => {
    const args = buildClaudeCompletionArgs("/tmp/empty-mcp.json", "claude-test")
    expect(args).toEqual([
      "-p",
      "--output-format",
      "text",
      "--permission-mode",
      "dontAsk",
      "--strict-mcp-config",
      "--mcp-config",
      "/tmp/empty-mcp.json",
      "--tools",
      "",
      "--model",
      "claude-test",
    ])
  })

  test("child environment keeps CLI auth paths but strips benchmark and provider secrets", () => {
    const env = buildCliEnvironment({
      PATH: "/bin",
      HOME: "/tmp/home",
      CODEX_HOME: "/tmp/codex",
      LANG: "en_US.UTF-8",
      OPENAI_API_KEY: "secret-openai",
      ANTHROPIC_API_KEY: "secret-anthropic",
      VOYAGE_API_KEY: "secret-voyage",
      SUPABASE_URL: "secret-url",
      HERMES_MB_LLM_CLI: "codex",
      CUSTOM_SECRET: "secret-custom",
      OPENAI_BASE_URL: "https://endpoint.invalid/v1",
      OPENAI_API_BASE: "https://endpoint.invalid/v1",
      AZURE_OPENAI_ENDPOINT: "https://endpoint.invalid",
    })
    expect(env).toMatchObject({
      PATH: "/bin",
      HOME: "/tmp/home",
      CODEX_HOME: "/tmp/codex",
      LANG: "en_US.UTF-8",
    })
    expect(env.OPENAI_API_KEY).toBeUndefined()
    expect(env.ANTHROPIC_API_KEY).toBeUndefined()
    expect(env.VOYAGE_API_KEY).toBeUndefined()
    expect(env.SUPABASE_URL).toBeUndefined()
    expect(env.HERMES_MB_LLM_CLI).toBeUndefined()
    expect(env.CUSTOM_SECRET).toBeUndefined()
    expect(env.OPENAI_BASE_URL).toBeUndefined()
    expect(env.OPENAI_API_BASE).toBeUndefined()
    expect(env.AZURE_OPENAI_ENDPOINT).toBeUndefined()
  })

  test("captures CLI version, pins, and provider usage without storing prompt or output", async () => {
    installFakeCodex()
    process.env.HERMES_MB_CODEX_MODEL = "gpt-test"
    process.env.HERMES_MB_CODEX_PROVIDER = "openai"
    process.env.HERMES_MB_CODEX_SERVICE_TIER = "priority"
    process.env.OPENAI_API_KEY = "must-not-reach-child"
    process.env.VOYAGE_API_KEY = "must-not-reach-child"
    let telemetry: CliCallTelemetry | undefined

    const text = await cliComplete("prompt must not be retained", {
      role: "answerer",
      effort: "medium",
      retry: false,
      onTelemetry: (value) => {
        telemetry = value
      },
    })

    expect(text).toBe("answer from fake CLI")
    expect(telemetry).toMatchObject({
      transportVersion: "codex-cli 1.2.3",
      requested: {
        model: "gpt-test",
        modelExplicit: true,
        reasoningEffort: "medium",
        provider: "openai",
        serviceTier: "priority",
      },
      usage: {
        inputTokens: 12,
        cachedInputTokens: 3,
        outputTokens: 4,
        reasoningOutputTokens: 2,
      },
      usageComplete: true,
      retryCount: 0,
    })
    expect(JSON.stringify(telemetry)).not.toContain("prompt must not be retained")
    expect(JSON.stringify(telemetry)).not.toContain("answer from fake CLI")
    expect(telemetry?.attempts).toHaveLength(1)
    expect(telemetry?.attempts[0]).toMatchObject({
      status: "completed",
      errorEventCount: 0,
      errorItemCount: 0,
    })
    expect(telemetry?.attempts[0]?.reroute).toBeUndefined()
  })

  test.each([
    {
      type: "item.completed",
      item: { type: "error", message: "model rerouted: gpt-6.1-sol -> other-model-1" },
    },
    { type: "error", message: "private error text" },
    { type: "turn.failed", error: { message: "private turn error" } },
  ])("fails exit-zero calls closed on $type; a reroute is not retried", async (event) => {
    installFakeCodex([event])
    process.env.HERMES_MB_CODEX_MODEL = "gpt-6.1-sol"
    let telemetry: CliCallTelemetry | undefined
    let returnedText: string | undefined
    let caught: unknown
    try {
      returnedText = await cliComplete("prompt must not be retained", {
        onTelemetry: (value) => {
          telemetry = value
        },
      })
    } catch (error) {
      caught = error
    }
    expect(caught).toBeInstanceOf(CliCallError)
    expect(returnedText).toBeUndefined()
    expect((caught as CliCallError).telemetry).toBe(telemetry!)
    expect(telemetry?.version).toBe("memorybench-cli-call-v1")
    const rerouted = event.type === "item.completed"
    expect(telemetry?.retryCount).toBe(rerouted ? 0 : 1)
    expect(telemetry?.usageComplete).toBe(false)
    expect(telemetry?.attempts).toHaveLength(rerouted ? 1 : 2)
    for (const attempt of telemetry!.attempts) {
      expect(attempt).toMatchObject({
        status: "failed",
        errorCode: "error_event",
        errorEventCount: event.type === "error" ? 1 : 0,
        errorItemCount: event.type === "error" ? 0 : 1,
      })
      if (event.type === "item.completed") {
        expect(attempt.reroute).toEqual({ requested: "gpt-6.1-sol", served: "other-model-1" })
      }
    }
    const stored = JSON.stringify(telemetry)
    expect(stored).not.toContain("prompt must not be retained")
    expect(stored).not.toContain("answer from fake CLI")
    expect(stored).not.toContain("private")
    expect(stored).not.toContain("model rerouted:")
  })
})

describe("structured CLI ledger", () => {
  test("counts error items, failed turns and top-level errors separately", () => {
    const parse = (events: unknown[]) =>
      parseCodexJsonlTelemetry(events.map((event) => JSON.stringify(event)).join("\n"))
    const message = {
      type: "item.completed",
      item: { type: "agent_message", text: "private answer" },
    }
    const errorItem = (type: string, message: string) => ({
      type,
      item: { type: "error", message },
    })
    const rerouted = parse([
      message,
      errorItem("item.completed", "model rerouted: gpt-6.1-sol -> other-model-1 private suffix"),
    ])
    expect(rerouted).toMatchObject({
      errorEventCount: 0,
      errorItemCount: 1,
      reroute: { requested: "gpt-6.1-sol", served: "other-model-1" },
    })
    expect(JSON.stringify(rerouted)).not.toContain("private")
    const normal = parse([message, message])
    expect(normal.errorItemCount).toBe(0)
    expect(normal.reroute).toBeUndefined()
    expect(parse([{ type: "turn.failed" }]).errorItemCount).toBe(1)
    const multiple = parse([
      errorItem("item.started", "model rerouted: first -> served-1"),
      errorItem("item.updated", "model rerouted: second -> served-2"),
      errorItem("item.completed", "other private error"),
      { type: "error", message: "private top-level error" },
    ])
    expect(multiple).toMatchObject({
      errorEventCount: 1,
      errorItemCount: 3,
      reroute: { requested: "first", served: "served-1" },
    })
  })

  test.each(["invalid/id", "x".repeat(65)])("redacts invalid reroute model ids: %s", (invalid) => {
    const parsed = parseCodexJsonlTelemetry(
      JSON.stringify({
        type: "item.completed",
        item: { type: "error", message: `model rerouted: ${invalid} -> ${invalid}` },
      })
    )
    expect(parsed.errorItemCount).toBe(1)
    expect(parsed.reroute).toEqual({ requested: "unparsed", served: "unparsed" })
    expect(JSON.stringify(parsed)).not.toContain(invalid)
  })

  test("parses only bounded provider telemetry and discloses mixed resume identities", () => {
    const parsed = parseCodexJsonlTelemetry(
      '{"type":"thread.started","thread_id":"t"}\n' +
        '{"type":"item.completed","text":"never persist this"}\n' +
        '{"type":"turn.completed","usage":{"input_tokens":8,"cached_input_tokens":2,"output_tokens":3,"reasoning_output_tokens":1}}\n'
    )
    expect(parsed).toMatchObject({
      threadId: "t",
      eventCount: 3,
      usage: {
        inputTokens: 8,
        cachedInputTokens: 2,
        outputTokens: 3,
        reasoningOutputTokens: 1,
      },
    })
    expect(JSON.stringify(parsed)).not.toContain("never persist this")

    const makeCall = (model: string): CliCallTelemetry => ({
      version: "memorybench-cli-call-v1",
      role: "answerer",
      transport: "codex-cli",
      transportVersion: "codex 1",
      requested: {
        model,
        modelExplicit: true,
        reasoningEffort: "medium",
        provider: "openai",
        serviceTier: "priority",
        pinSource: "explicit-cli-argv",
      },
      eventModelField: "not-emitted-by-codex-jsonl",
      attempts: [],
      usage: {
        inputTokens: 1,
        cachedInputTokens: 0,
        outputTokens: 1,
        reasoningOutputTokens: 0,
      },
      usageComplete: true,
      totalDurationMs: 1,
      retryCount: 0,
    })
    const ledger = summarizeCliLedger([
      { status: "completed", llmCalls: [makeCall("model-a")] },
      { status: "completed", llmCalls: [makeCall("model-b")] },
    ])
    expect(ledger).toMatchObject({
      callCount: 2,
      executionIdentityCount: 2,
      mixedExecutionIdentity: true,
      callLedgerComplete: true,
    })
  })
})
