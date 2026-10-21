import { afterEach, expect, test } from "bun:test"
import { assertSafeHarnessModes } from "./mode-guard"

const keys = ["HERMES_MB_ALLOW_UNSAFE_1700", "HERMES_MB_LLM_CLI",
  "HERMES_MB_ANSWER_PRESENTATION", "HERMES_MB_DETERMINISTIC_OPERATIONS"]
const saved = Object.fromEntries(keys.map((key) => [key, process.env[key]]))
afterEach(() => {
  for (const key of keys) {
    if (saved[key] === undefined) delete process.env[key]
    else process.env[key] = saved[key]
  }
})

test("deferred modes refuse before execution and accept only the explicit override", () => {
  for (const key of keys) delete process.env[key]
  for (const [key, value] of [[keys[1]!, "claude"], [keys[2]!, "evidence_cards_v1"], [keys[3]!, "1"]]) {
    process.env[key] = value
    expect(() => assertSafeHarnessModes()).toThrow("#1700")
    process.env.HERMES_MB_ALLOW_UNSAFE_1700 = "1"
    expect(() => assertSafeHarnessModes()).not.toThrow()
    delete process.env.HERMES_MB_ALLOW_UNSAFE_1700
    delete process.env[key]
  }
  expect(() => assertSafeHarnessModes({ answeringModel: "stored" })).toThrow("#1700")
  expect(() => assertSafeHarnessModes({ answeringModel: "stored" }, "changed")).toThrow("#1700")
  for (const phase of ["evaluate", "report"]) {
    expect(() => assertSafeHarnessModes(undefined, undefined, phase)).toThrow("#1700")
  }
  expect(() => assertSafeHarnessModes({ answeringModel: "stored" }, "stored")).not.toThrow()
  process.env.HERMES_MB_LLM_CLI = "codex"
  expect(() => assertSafeHarnessModes({ answeringModel: "stored" })).not.toThrow()
  expect(() => assertSafeHarnessModes()).not.toThrow()
})
