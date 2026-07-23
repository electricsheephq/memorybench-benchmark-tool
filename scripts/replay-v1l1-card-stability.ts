import { readFileSync } from "node:fs"
import { renderEvidenceCards } from "../src/prompts/evidence-cards"

const checkpointPath =
  process.env.V1L1_FROZEN_CHECKPOINT ||
  "/Volumes/LEXAR/Codex/session-notes/2026-07-21/hermes-lcm-autonomous-release-program/artifacts/m450/checkpoint.json"
const checkpoint = JSON.parse(readFileSync(checkpointPath, "utf8")) as {
  questions: Record<string, { phases: { search: { results: unknown[] } } }>
}
const sampleIds = [
  "gpt4_468eb063",
  "9ee3ecd6",
  "gpt4_a56e767c",
  "gpt4_7abb270c",
  "eaca4986",
  "2b8f3739",
  "d851d5ba",
  "09ba9854",
  "92a0aa75",
  "d6062bb9",
]
const cases = sampleIds.map((questionId) => {
  const context = checkpoint.questions[questionId]?.phases.search.results
  if (!context) throw new Error(`frozen context missing for ${questionId}`)
  const first = renderEvidenceCards(context)
  const second = renderEvidenceCards(structuredClone(context))
  return {
    questionId,
    byteStable:
      first.text === second.text &&
      first.provenance.renderedTextSha256 === second.provenance.renderedTextSha256,
    renderedTextSha256: first.provenance.renderedTextSha256,
    renderedItems: first.provenance.renderedItems,
  }
})
const output = {
  suite: "V1-L1 evidence-card byte-stability",
  samples: cases.length,
  passed: cases.every((item) => item.byteStable),
  cases,
}
console.log(JSON.stringify(output, null, 2))
if (!output.passed) process.exitCode = 1
