import { describe, expect, test } from "bun:test"
import { LOCOMO_QUESTION_TYPES } from "../benchmarks/locomo"
import {
  ABSTENTION_JUDGE_PROMPT,
  DEFAULT_JUDGE_PROMPT,
  TEMPORAL_JUDGE_PROMPT,
} from "../prompts/defaults"
import { getJudgePrompt } from "./base"

describe("LoCoMo judge category routing", () => {
  test.each([
    ["adversarial", ABSTENTION_JUDGE_PROMPT],
    ["temporal", TEMPORAL_JUDGE_PROMPT],
    ["single-hop", DEFAULT_JUDGE_PROMPT],
    ["multi-hop", DEFAULT_JUDGE_PROMPT],
    ["world-knowledge", DEFAULT_JUDGE_PROMPT],
  ])("%s selects the expected judge prompt", (questionType, expectedPrompt) => {
    expect(LOCOMO_QUESTION_TYPES[questionType]).toBeDefined()
    expect(getJudgePrompt(questionType)).toBe(expectedPrompt)
  })

  test.each(["single_hop", "multi_hop", "world"])(
    "%s spelling remains on the default LoCoMo route",
    (questionType) => {
      expect(getJudgePrompt(questionType)).toBe(DEFAULT_JUDGE_PROMPT)
    }
  )

  test("adversarial prompt credits premise rejection without vouching for elaboration", () => {
    expect(ABSTENTION_JUDGE_PROMPT).toContain("correctly abstains")
    expect(ABSTENTION_JUDGE_PROMPT).toContain("rejects the question's false premise")
    // The judge has no conversation/evidence, so it must not certify the
    // accuracy of extra asserted details -- only contradiction is scoreable.
    expect(ABSTENTION_JUDGE_PROMPT).toContain("do not credit or penalize extra elaboration")
    expect(ABSTENTION_JUDGE_PROMPT).toContain("contradicts the provided correct answer")
  })
})
