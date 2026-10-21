import type { RunCheckpoint } from "../types/checkpoint"
import { cliLlmBackend } from "../utils/cli-llm"

/** Deferred #1700 modes require an explicit opt-in until their repairs land. */
export function assertSafeHarnessModes(
  checkpoint?: Pick<RunCheckpoint, "answeringModel" | "answerPresentationMode"> | null,
  answeringModel?: string,
  copyPhase?: string
): void {
  if (process.env.HERMES_MB_ALLOW_UNSAFE_1700 === "1") return
  const backend = cliLlmBackend()
  const mode = checkpoint?.answerPresentationMode || process.env.HERMES_MB_ANSWER_PRESENTATION
  const reasons = [
    process.env.HERMES_MB_DETERMINISTIC_OPERATIONS === "1" && "deterministic-operation selector",
    mode === "evidence_cards_v1" && "evidence-card supplied refs",
    backend === "claude" && "Claude reader UTF-8/OAuth",
    checkpoint && !backend && answeringModel !== checkpoint.answeringModel && "ai-sdk resume without matching model",
    (copyPhase === "evaluate" || copyPhase === "report") && "advanced evaluate/report copy",
  ].filter(Boolean)
  if (reasons.length) throw new Error(
    `Refusing deferred #1700 mode(s): ${reasons.join(", ")}. Explicit override: HERMES_MB_ALLOW_UNSAFE_1700=1 (CLI: --allow-unsafe-1700).`
  )
}
