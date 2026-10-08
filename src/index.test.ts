import { describe, expect, test } from "bun:test"
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { tmpdir } from "node:os"

describe("entry point", () => {
  test("a run that throws exits non-zero (lcm-x #976)", () => {
    const dir = mkdtempSync(join(tmpdir(), "memorybench-exit-"))
    try {
      // An unreadable checkpoint makes the continued run throw before any provider or model call.
      mkdirSync(join(dir, "data", "runs", "broken"), { recursive: true })
      writeFileSync(join(dir, "data", "runs", "broken", "checkpoint.json"), "{ not json")
      const result = Bun.spawnSync(
        [process.execPath, "run", join(import.meta.dir, "index.ts"), "run", "-r", "broken"],
        { cwd: dir, stdout: "pipe", stderr: "pipe", env: { ...process.env } }
      )
      expect(result.exitCode).toBe(1)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
