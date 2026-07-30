import { afterEach, describe, expect, test } from "bun:test"
import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { join, relative } from "node:path"
import { LoCoMoBenchmark } from "."

const fixtureDirs: string[] = []

function writeFixture(): string {
  const fixtureDir = mkdtempSync(join(process.cwd(), "data", "locomo-fixture-"))
  fixtureDirs.push(fixtureDir)
  const fixturePath = join(fixtureDir, "locomo.json")
  writeFileSync(
    fixturePath,
    JSON.stringify([
      {
        sample_id: "fixture",
        conversation: {
          speaker_a: "A",
          speaker_b: "B",
          session_1_date_time: "1:00 pm on 1 January, 2024",
          session_1: [
            {
              speaker: "A",
              dia_id: "D1:1",
              text: "Look what I made.",
              blip_caption: "a red kite with a blue tail",
              img_url: ["https://example.test/kite.jpg"],
            },
          ],
        },
        qa: [
          {
            question: "What did A make?",
            answer: "a red kite with a blue tail",
            evidence: ["D1:1"],
            category: 1,
          },
          {
            question: "What false claim was made?",
            adversarial_answer: "B made the kite",
            evidence: ["D1:1"],
            category: 5,
          },
          {
            question: "Did B make the kite?",
            answer: "No",
            adversarial_answer: "Yes",
            evidence: ["D1:1"],
            category: 5,
          },
        ],
        event_summary: {},
        observation: {},
        session_summary: {},
      },
    ])
  )
  return relative(process.cwd(), fixturePath)
}

afterEach(() => {
  for (const fixtureDir of fixtureDirs.splice(0)) {
    rmSync(fixtureDir, { recursive: true, force: true })
  }
})

describe("LoCoMo fixture ingestion", () => {
  test("ingests caption-only gold evidence and notes the image URL", async () => {
    const benchmark = new LoCoMoBenchmark()
    await benchmark.load({ dataPath: writeFixture() })

    const content = benchmark.getHaystackSessions("fixture-q0")[0]?.messages[0]?.content
    expect(content).toBe("Look what I made. [shared image: a red kite with a blue tail]")
    // URLs never enter evidence: descriptive filenames can leak gold labels.
    expect(content).not.toContain("img_url")
    expect(content).not.toContain("https://")
  })

  test("adversarial-only rows use the canonical abstention gold, never the trap", async () => {
    const benchmark = new LoCoMoBenchmark()
    await benchmark.load({ dataPath: writeFixture() })

    expect(benchmark.getGroundTruth("fixture-q1")).toBe("Not mentioned in the conversation")
    // The trap completion must never be presented to the judge as gold.
    expect(benchmark.getGroundTruth("fixture-q1")).not.toBe("B made the kite")
    expect(benchmark.getGroundTruth("fixture-q1")).not.toBe("undefined")
  })

  test("prefers the explicit answer when an adversarial row has both fields", async () => {
    const benchmark = new LoCoMoBenchmark()
    await benchmark.load({ dataPath: writeFixture() })

    expect(benchmark.getGroundTruth("fixture-q2")).toBe("No")
  })
})
