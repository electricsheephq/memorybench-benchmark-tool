import { describe, expect, test } from "bun:test"
import {
  appendDeterministicTrace,
  parseDeterministicOperationRequest,
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
