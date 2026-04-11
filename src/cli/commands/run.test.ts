import { describe, expect, test } from "bun:test"
import { parseRunArgs } from "./run"

describe("parseRunArgs", () => {
  test("parses --questions into explicit question IDs", () => {
    const parsed = parseRunArgs([
      "-p",
      "cortex",
      "-b",
      "locomo",
      "--questions",
      "q0,q82,q152",
    ])

    expect(parsed).toBeTruthy()
    expect(parsed?.questions).toEqual(["q0", "q82", "q152"])
    expect(parsed?.sample).toBeUndefined()
  })
})
