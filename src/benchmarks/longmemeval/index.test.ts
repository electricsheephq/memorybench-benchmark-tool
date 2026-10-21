import { expect, test } from "bun:test"
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs"
import { createHash } from "node:crypto"
import { tmpdir } from "node:os"
import { join, relative } from "node:path"
import { LongMemEvalBenchmark } from "."

test("raw-source loader retains base filename enumeration, including prefix ids", async () => {
  const root = mkdtempSync(join(tmpdir(), "lme-order-"))
  try {
    const ids = ["a0", "a", "a_abs", "b", "10", "1", "c"]
    const raw = JSON.stringify(
      ids.map((question_id, i) => ({
        question_id,
        question: "Synthetic",
        answer: "Synthetic",
        question_type: i % 2 ? "multi-session" : "single-session-user",
        haystack_sessions: [],
        haystack_dates: [],
      }))
    )
    writeFileSync(join(root, "longmemeval_s_cleaned.json"), raw)
    const benchmark = new LongMemEvalBenchmark()
    await benchmark.load({ dataPath: relative(process.cwd(), root) })
    const base = readdirSync(join(root, "questions"))
      .filter((f) => f.endsWith(".json"))
      .map((file) => JSON.parse(readFileSync(join(root, "questions", file), "utf8")).question_id)
    expect(benchmark.getQuestions().map((q) => q.questionId)).toEqual(base)
    expect(benchmark.getQuestions({ limit: 3 }).map((q) => q.questionId)).toEqual(base.slice(0, 3))
    expect(benchmark.datasetSha256).toBe(createHash("sha256").update(raw).digest("hex"))
  } finally {
    rmSync(root, { recursive: true })
  }
})
