import { describe, expect, test } from "bun:test"
import { buildAnswerPrompt } from "../../orchestrator/phases/answer"
import { getProviderConfig } from "../../utils/config"
import { HermesLcmProvider, normalizeHermesSearchResponse } from "./index"

describe("Hermes-LCM ordinary provider contract", () => {
  test("passes the producer's speaker and session dates unchanged to the bridge", async () => {
    const provider = new HermesLcmProvider()
    const session = {
      sessionId: "fixture-session",
      messages: [
        { role: "user" as const, content: "First turn", speaker: "Alpha" },
        { role: "assistant" as const, content: "Second turn", speaker: "Beta" },
      ],
      metadata: { date: "2023-05-30T18:09:00.000Z", formattedDate: "6:09 pm on 30 May, 2023" },
    }
    const requests: unknown[] = []
    const state = provider as unknown as { handles: Map<string, unknown> }
    state.handles.set("fixture", {
      request: async (payload: unknown) => {
        requests.push(payload)
        return { ok: true, documentIds: ["1", "2"] }
      },
    })
    expect(await provider.ingest([session], { containerTag: "fixture" })).toEqual({
      documentIds: ["1", "2"],
    })
    expect(requests).toEqual([{ cmd: "ingest", containerTag: "fixture", session }])
  })

  test("uses the bridge-owned provider configuration without a harness API key", () => {
    expect(getProviderConfig("hermes-lcm")).toEqual({ apiKey: "" })
  })

  test("unwraps bridge results byte-for-byte and excludes provenance from the answer prompt", () => {
    const results = [
      {
        content: "The exact ordinary answer-ready result.",
        metadata: { session_id: "session-7", date: "2023-05-30T00:00:00.000Z" },
      },
    ]
    const provenanceSentinel = "PROVENANCE_MUST_NOT_REACH_SCORED_CONTEXT"
    const bridgeResponse = {
      ok: true,
      results,
      provenance: { audit: provenanceSentinel },
      degraded: false,
    }

    const normalized = normalizeHermesSearchResponse(bridgeResponse)

    expect(normalized).toBe(results)
    expect(JSON.stringify(normalized, null, 2)).toBe(JSON.stringify(results, null, 2))

    const provider = new HermesLcmProvider()
    const prompt = buildAnswerPrompt(
      "What was remembered?",
      normalized,
      "2023/05/31 (Wed) 12:00",
      provider
    )
    const answerPrompt = provider.prompts.answerPrompt
    if (typeof answerPrompt !== "function") throw new Error("Expected function answer prompt")
    const historicalNormalizedPrompt = answerPrompt(
      "What was remembered?",
      results,
      "2023/05/31 (Wed) 12:00"
    )

    expect(prompt).toBe(historicalNormalizedPrompt)
    expect(prompt).toContain(JSON.stringify(results, null, 2))
    expect(prompt).not.toContain(provenanceSentinel)
  })
})
