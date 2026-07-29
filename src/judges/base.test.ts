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
})
