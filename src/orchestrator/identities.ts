import { createHash } from "node:crypto"
import { isAbsolute, win32 } from "node:path"
import type { Benchmark, DatasetIdentity } from "../types/benchmark"
import type { RunCheckpoint, RunIdentities } from "../types/checkpoint"
import type { Provider } from "../types/provider"
import * as prompts from "../prompts/defaults"
import { SCORING_VERSION } from "../judges/base"

export function relativeDatasetPath(datasetPath: string): string {
  if (isAbsolute(datasetPath) || win32.isAbsolute(datasetPath)) {
    throw new Error("datasetPath must be a relative label")
  }
  return datasetPath
}

export function datasetIdentity(datasetPath: string, bytes: Uint8Array): DatasetIdentity {
  return {
    datasetPath: relativeDatasetPath(datasetPath),
    datasetSha256: createHash("sha256").update(bytes).digest("hex"),
  }
}

export function runIdentities(
  benchmark: Benchmark,
  provider: Pick<Provider, "name" | "prompts">,
  judge: string
): RunIdentities {
  if (!benchmark.datasetPath || !benchmark.datasetSha256)
    throw new Error("Dataset identity not loaded")
  const override = provider.prompts?.judgePrompt
  // Export names sorted lexically define the fixed order of all prompt constants.
  const texts = Object.entries(prompts)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .filter(([, value]) => typeof value === "string")
    .map(([, value]) => value as string)
  return {
    datasetPath: benchmark.datasetPath,
    datasetSha256: benchmark.datasetSha256,
    promptVersion: `sha256:${createHash("sha256")
      .update(texts.join("") + (override?.toString() || ""))
      .digest("hex")}`,
    scoringPath: `${judge}/${override ? `provider:${provider.name}` : "default"}/${SCORING_VERSION}`,
    answerPresentationMode: "raw_json_v1",
  }
}

export function assertResumeIdentities(checkpoint: RunCheckpoint, current: RunIdentities): void {
  for (const key of ["datasetSha256", "promptVersion", "scoringPath"] as const) {
    if (checkpoint[key] !== undefined && checkpoint[key] !== current[key]) {
      throw new Error(`Resume refused: identity mismatch: ${key}`)
    }
  }
}
