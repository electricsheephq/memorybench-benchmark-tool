import type { ProviderPrompts } from "../../types/prompts"
import { buildDefaultAnswerPrompt } from "../../prompts/defaults"

/**
 * FIX B (MB2 forensics rerun): hermes-lcm keeps the HARNESS-DEFAULT answer prompt
 * PLUS a small, provider-neutral reasoning rider that targets two forensically
 * attributed answerer failures — temporal arithmetic (27 failures) and
 * knowledge-update latest-fact selection. The rider adds NO dataset-specific
 * knowledge and NO evidence peeking: it only tells the answerer to (a) compute
 * elapsed-time answers from the ISO dates the provider already surfaces in each
 * hit's metadata rather than reading "today"/"yesterday" out of memory text, and
 * (b) prefer the most recent memory when facts conflict. The JUDGE prompt stays
 * undefined so it falls through to LongMemEval's standard per-question-type
 * prompts (`getJudgePromptForType`) — the judge was exonerated by the forensics.
 */
const REASONING_RIDER = `
- Each memory in the context carries an ISO date in its metadata ("date"). When the question asks how long ago / since / between / for how long, COMPUTE the interval from those metadata dates against the Question Date above — do NOT read "today", "yesterday", or "now" literally out of the memory text.
- When facts conflict across memories (a knowledge update), prefer the MOST RECENT memory by its metadata date.`

export const HERMES_LCM_PROMPTS: ProviderPrompts = {
  answerPrompt: (question: string, context: unknown[], questionDate?: string): string => {
    // Build on the harness default so it stays the single source of truth, then
    // splice the rider in just before the final "Answer:" cue.
    const base = buildDefaultAnswerPrompt(question, context, questionDate)
    return base.replace(/\n\nAnswer:$/, `${REASONING_RIDER}\n\nAnswer:`)
  },
}
