import { describe, expect, test } from "bun:test"
import {
  appendDeterministicTrace,
  inferOperation,
  parseDeterministicOperationRequest,
  shouldAttemptDeterministicOperation,
  validateDeterministicOperation,
} from "./deterministic-operations"

const evidence = [
  {
    content: "The taxi cost $60.",
    metadata: { session_id: "a", date: "2023-03-01", role: "user", exact_ref: "lcm:1:0-19" },
  },
  {
    content: "The train cost $10.",
    metadata: { session_id: "b", date: "2023-03-02", role: "user", exact_ref: "lcm:2:0-20" },
  },
  {
    content: "I attended the Alpha workshop.",
    metadata: { session_id: "c", date: "2023-03-03", role: "user", exact_ref: "lcm:3:0-30" },
  },
  {
    content: "I mentioned the Alpha workshop again.",
    metadata: { session_id: "d", date: "2023-03-04", role: "user", exact_ref: "lcm:4:0-36" },
  },
]

describe("deterministic operation layer", () => {
  test("computes only exact cited typed operands", () => {
    const decision = validateDeterministicOperation(
      {
        operation: "difference",
        direction: "first_minus_second",
        operands: [
          { evidenceIndex: 0, exactRef: "lcm:1:0-19", quote: "$60", value: 60, unit: "usd" },
          { evidenceIndex: 1, exactRef: "lcm:2:0-20", quote: "$10", value: 10, unit: "usd" },
        ],
      },
      evidence,
      "How much did I save?"
    )
    expect(decision.status).toBe("computed")
    if (decision.status === "computed") {
      expect(decision.trace.resultValue).toBe(50)
      expect(decision.trace.operandProvenance).toHaveLength(2)
      expect(appendDeterministicTrace("Question\n\nAnswer:", decision.trace)).toContain("50 usd")
    }
  })
  test("normalizes selector quote and dash transport before exact source matching", () => {
    const unicodeEvidence = [
      {
        content: "The \u201cfirst\u2011order\u201d discount was \uff14\uff10\uff05.",
        metadata: {
          session_id: "unicode",
          date: "2023-03-01",
          role: "user",
          exact_ref: "lcm:5:0-41",
        },
      },
    ]
    const decision = validateDeterministicOperation(
      {
        operation: "sum",
        operands: [
          {
            evidenceIndex: 0,
            exactRef: "lcm:5:0-41",
            quote: 'The "first-order" discount was 40%.',
            value: 40,
            unit: "percent",
          },
        ],
      },
      unicodeEvidence,
      "What was the total discount?"
    )
    expect(decision).toMatchObject({ status: "computed", trace: { resultValue: 40 } })
  })
  test("treats compared-to-first phrasing as a comparison, not an ordinal", () => {
    expect(
      inferOperation(
        "Did I receive a higher percentage discount on my first order from HelloFresh, compared to my first UberEats order?"
      )
    ).toBe("difference")
    expect(
      inferOperation("How much earlier do I wake up on Fridays compared to other weekdays?")
    ).toBe("difference")
    expect(inferOperation("Which song was second?")).toBe("ordinal")
  })
  test("deduplicates explicit canonical count keys", () => {
    const decision = validateDeterministicOperation(
      {
        operation: "count",
        operands: [
          { evidenceIndex: 2, exactRef: "lcm:3:0-30", quote: "Alpha workshop", key: "alpha" },
          { evidenceIndex: 3, exactRef: "lcm:4:0-36", quote: "Alpha workshop", key: "alpha" },
        ],
      },
      evidence,
      "How many workshops did I attend?"
    )
    expect(decision.status).toBe("computed")
    if (decision.status === "computed") expect(decision.trace.resultValue).toBe(1)
  })
  test("uses plain-reading fallback for forged, mismatched, or malformed operands", () => {
    const forged = validateDeterministicOperation(
      {
        operation: "sum",
        operands: [
          { evidenceIndex: 0, exactRef: "lcm:1:0-19", quote: "$60", value: 600, unit: "usd" },
        ],
      },
      evidence,
      "What was the total cost?"
    )
    expect(forged).toMatchObject({ status: "fallback" })
    expect(parseDeterministicOperationRequest("not json")).toBeUndefined()
    expect(parseDeterministicOperationRequest('{"operation":"sum","operands":[]}')?.operation).toBe(
      "sum"
    )
  })
  test("engages by default in card mode whenever the question requires an operation", () => {
    // The V1L1-LOSS8 regression: count/date questions in card mode must not
    // stay `not_attempted` just because no env opt-in was exported.
    const opQuestions = [
      "How many health-related devices do I use in a day?",
      "What was the page count of the two novels I finished in January and March?",
      "How many days ago did I meet Emma?",
      "What is the chronological order of the six events?",
      "Which song was second?",
      "What was the total raised by March 20?",
      "How much would I save by taking the train?",
    ]
    for (const question of opQuestions) {
      expect(shouldAttemptDeterministicOperation(question, "evidence_cards_v1", undefined)).toBe(
        true
      )
    }
    // No operation required -> no selector call (and no extra LLM spend).
    expect(
      shouldAttemptDeterministicOperation(
        "Where do I currently keep my old sneakers?",
        "evidence_cards_v1",
        undefined
      )
    ).toBe(false)
  })
  test("honors presentation mode and env force/kill overrides", () => {
    const question = "How many workshops did I attend?"
    // Never outside card mode: operand citations require card exact refs.
    expect(shouldAttemptDeterministicOperation(question, "raw_json_v1", undefined)).toBe(false)
    expect(shouldAttemptDeterministicOperation(question, "raw_json_v1", "1")).toBe(false)
    // "0" is a kill switch, "1" forces attempts even without an inferred op.
    expect(shouldAttemptDeterministicOperation(question, "evidence_cards_v1", "0")).toBe(false)
    expect(
      shouldAttemptDeterministicOperation("What did I say about Denver?", "evidence_cards_v1", "1")
    ).toBe(true)
  })
  test("calculates date differences from exactly two cited canonical dates", () => {
    const decision = validateDeterministicOperation(
      {
        operation: "date_diff_days",
        operands: [
          { evidenceIndex: 0, exactRef: "lcm:1:0-19", quote: "taxi", date: "2023-03-01" },
          { evidenceIndex: 1, exactRef: "lcm:2:0-20", quote: "train", date: "2023-03-10" },
        ],
      },
      evidence,
      "How many days between the events?"
    )
    expect(decision).toMatchObject({ status: "computed", trace: { resultValue: 9 } })
  })
})
