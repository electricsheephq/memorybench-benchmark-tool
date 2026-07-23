import { describe, expect, test } from "bun:test"
import { createHash } from "node:crypto"
import {
  EVIDENCE_CARD_READER_CONTRACT,
  EVIDENCE_CARD_READER_CONTRACT_SHA256,
  renderEvidenceCards,
} from "./evidence-cards"

const sourceContext = [
  {
    content: "I moved to Denver in April.",
    metadata: {
      session_id: "private-source-session-one",
      date: "2024-04-06T09:30:00.000Z",
      role: "user",
      exact_ref: "lcm:7:0-27",
      api_key: "must-never-be-rendered",
    },
  },
  {
    content: "You said Austin was your previous home.",
    metadata: {
      session_id: "private-source-session-two",
      date: "2024-04-07T10:00:00.000Z",
      role: "assistant",
      store_id: 9,
      content_offset: 4,
      content_returned_chars: 39,
    },
  },
  {
    content: "Denver is now home.",
    metadata: {
      session_id: "private-source-session-one",
      date: "2024-04-08T12:00:00.000Z",
      role: "user",
      exact_ref: "lcm:11:0-19",
    },
  },
]

describe("Hermes-LCM evidence-card presentation", () => {
  test("renders every source item exactly once with opaque sessions and stable exact refs", () => {
    const rendered = renderEvidenceCards(sourceContext)
    expect(rendered.provenance.inputItems).toBe(3)
    expect(rendered.provenance.renderedItems).toBe(3)
    expect(rendered.text).toContain("[lcm:7:0-27 | user]")
    expect(rendered.text).toContain("[lcm:9:4-43 | assistant]")
    expect(rendered.text).not.toContain("private-source-session")
    expect(rendered.text).not.toContain("must-never-be-rendered")
    expect(rendered.text.match(/SESSION session-[0-9a-f]{12}/g)).toHaveLength(2)
  })
  test("is byte-stable and attests exact input and rendered content", () => {
    const first = renderEvidenceCards(sourceContext)
    const second = renderEvidenceCards(structuredClone(sourceContext))
    expect(second).toEqual(first)
    expect(first.provenance.renderedTextSha256).toBe(
      createHash("sha256").update(first.text, "utf8").digest("hex")
    )
    expect(EVIDENCE_CARD_READER_CONTRACT_SHA256).toBe(
      createHash("sha256").update(EVIDENCE_CARD_READER_CONTRACT, "utf8").digest("hex")
    )
  })
  test("fails closed without a validated exact source span", () => {
    expect(() =>
      renderEvidenceCards([
        {
          content: "A fact without source coordinates.",
          metadata: { session_id: "session", date: "2024-04-06T00:00:00.000Z", role: "user" },
        },
      ])
    ).toThrow("exact source reference")
  })
})
