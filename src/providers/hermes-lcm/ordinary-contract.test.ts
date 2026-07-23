import { describe, expect, test } from "bun:test"
import { buildAnswerPrompt } from "../../orchestrator/phases/answer"
import { getProviderConfig } from "../../utils/config"
import { HermesLcmProvider, normalizeHermesSearchResponse } from "./index"

describe("Hermes-LCM ordinary provider contract", () => {
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
