import type { RunCheckpoint } from "../types/checkpoint"
import type { UnifiedQuestion } from "../types/unified"

export function questionCheckpointMetadata(question: UnifiedQuestion): {
  question: string
  groundTruth: string
  questionType: string
  questionDate?: string
} {
  const rawQuestionDate = question.metadata?.questionDate
  return {
    question: question.question,
    groundTruth: question.groundTruth,
    questionType: question.questionType,
    questionDate: typeof rawQuestionDate === "string" ? rawQuestionDate : undefined,
  }
}

/** Backfill legacy checkpoints without overwriting a date already persisted. */
export function syncQuestionCheckpointMetadata(
  checkpoint: RunCheckpoint,
  questions: UnifiedQuestion[]
): number {
  let updated = 0
  for (const question of questions) {
    const existing = checkpoint.questions[question.questionId]
    if (!existing || existing.questionDate) continue
    const { questionDate } = questionCheckpointMetadata(question)
    if (questionDate) {
      existing.questionDate = questionDate
      updated++
    }
  }
  return updated
}
