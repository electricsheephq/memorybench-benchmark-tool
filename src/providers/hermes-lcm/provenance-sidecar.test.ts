import { afterEach, describe, expect, test } from "bun:test"
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { HermesLcmProvider } from "./index"

const roots: string[] = []
const containerTag = "fixture-container"

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function fixture(response: Record<string, unknown>) {
  const root = mkdtempSync(join(tmpdir(), "hermes-provenance-"))
  roots.push(root)
  const provider = new HermesLcmProvider()
  const state = provider as unknown as { workdir: string; handles: Map<string, unknown> }
  state.workdir = root
  state.handles.set(containerTag, { request: async () => response })
  return { provider, path: join(root, "recall-provenance.jsonl") }
}

describe("Hermes-LCM durable recall provenance", () => {
  test("appends one content-free line and returns unchanged results", async () => {
    const query = "QUERY_MUST_NOT_BE_RECORDED"
    const results = [{ content: "CONTENT_MUST_NOT_BE_RECORDED" }]
    const coverage = { fts: "ok", nested: { preserved: true } }
    const harness_settings = {
      HERMES_MB_EVENT_TIME: "session",
      HERMES_MB_SENDER_RENDER: "gateway",
    }
    const { provider, path } = fixture({
      ok: true,
      results,
      degraded: true,
      degraded_reason: "embeddings_disabled",
      provenance: { coverage, harness_settings, unparsed_session_dates: 2,
        process_harness_settings: { HERMES_MB_EVENT_TIME: "off", HERMES_MB_SENDER_RENDER: "off" },
        other: "PROVENANCE_CONTENT_MUST_NOT_BE_RECORDED" },
    })

    expect(await provider.search(query, { containerTag })).toBe(results)
    const text = readFileSync(path, "utf8")
    const lines = text.trimEnd().split("\n")
    expect(lines).toHaveLength(1)
    const row = JSON.parse(lines[0]!)
    expect(row).toEqual({
      ts: expect.any(String),
      containerTag,
      degraded: true,
      degraded_reason: "embeddings_disabled",
      coverage,
      harness_settings,
      process_harness_settings: { HERMES_MB_EVENT_TIME: "off", HERMES_MB_SENDER_RENDER: "off" },
      unparsed_session_dates: 2,
      result_count: 1,
    })
    expect(new Date(row.ts).toISOString()).toBe(row.ts)
    expect(text.endsWith("\n")).toBe(true)
    expect(text).not.toContain(query)
    expect(text).not.toContain(results[0]!.content)
    expect(text).not.toContain("PROVENANCE_CONTENT_MUST_NOT_BE_RECORDED")
  })

  test("defaults missing provenance and degraded fields and appends each search", async () => {
    const { provider, path } = fixture({ ok: true, results: [] })
    for (let i = 0; i < 2; i++) await provider.search("query", { containerTag })
    const lines = readFileSync(path, "utf8").trimEnd().split("\n")
    expect(lines).toHaveLength(2)
    for (const line of lines) {
      expect(JSON.parse(line)).toEqual({
        ts: expect.any(String),
        containerTag,
        degraded: false,
        degraded_reason: null,
        coverage: null,
        harness_settings: {
          HERMES_MB_EVENT_TIME: "off",
          HERMES_MB_SENDER_RENDER: "off",
        },
        unparsed_session_dates: 0,
        result_count: 0,
      })
    }
  })

  test("rejects the search if the provenance append fails", async () => {
    const { provider, path } = fixture({ ok: true, results: [] })
    mkdirSync(path)
    await expect(provider.search("query", { containerTag })).rejects.toThrow()
  })

  test("records the container total from ingest, including a resumed response", async () => {
    const { provider, path } = fixture({
      ok: true, documentIds: ["1"], unparsed_session_dates: 1,
      unparsed_session_dates_total: 3, resumed: true,
    })
    const sessions = [{ sessionId: "synthetic", messages: [{ role: "user" as const, content: "PRIVATE_FIXTURE" }] }]
    for (let i = 0; i < 2; i++) {
      expect(await provider.ingest(sessions, { containerTag })).toEqual({ documentIds: ["1"] })
    }
    const text = readFileSync(join(path, "..", "ingest-provenance.jsonl"), "utf8")
    const rows = text.trimEnd().split("\n").map(line => JSON.parse(line))
    expect(rows).toHaveLength(2)
    expect(rows.map(row => row.unparsed_session_dates)).toEqual([3, 3])
    expect(rows.every(row => row.containerTag === containerTag)).toBe(true)
    expect(text).not.toContain("PRIVATE_FIXTURE")
  })
})
