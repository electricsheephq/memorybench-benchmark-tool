import type { Judge } from "../../types/judge"
import type { Benchmark } from "../../types/benchmark"
import type { RunCheckpoint } from "../../types/checkpoint"
import type { Provider } from "../../types/provider"
import { CheckpointManager } from "../checkpoint"
import { logger } from "../../utils/logger"
import { ConcurrentExecutor } from "../concurrent"
import { resolveConcurrency } from "../../types/concurrency"
import { calculateRetrievalMetrics } from "./retrieval-eval"
import {
  cliCallTelemetryFromError,
  cliCallsFromPhase,
  cliLlmBackend,
  reconcileCliProvenanceIdentity,
  summarizeCliLedger,
  type CliCallTelemetry,
} from "../../utils/cli-llm"

export async function runEvaluatePhase(
  judge: Judge,
  benchmark: Benchmark,
  checkpoint: RunCheckpoint,
  checkpointManager: CheckpointManager,
  questionIds?: string[],
  provider?: Provider
): Promise<void> {
  const questions = benchmark.getQuestions()
  const targetQuestions = questionIds
    ? questions.filter((q) => questionIds.includes(q.questionId))
    : questions

  const pendingQuestions = targetQuestions.filter((q) => {
    const status = checkpointManager.getPhaseStatus(checkpoint, q.questionId, "evaluate")
    const answerStatus = checkpointManager.getPhaseStatus(checkpoint, q.questionId, "answer")
    const hypothesis = checkpoint.questions[q.questionId]?.phases.answer.hypothesis
    return status !== "completed" && answerStatus === "completed" && hypothesis
  })

  if (pendingQuestions.length === 0) {
    updateJudgeCliLedger(checkpoint, checkpointManager)
    logger.info("No questions pending evaluation")
    return
  }

  const concurrency = resolveConcurrency("evaluate", checkpoint.concurrency, provider?.concurrency)

  logger.info(
    `Evaluating ${pendingQuestions.length} questions with ${judge.name} (concurrency: ${concurrency})...`
  )

  await ConcurrentExecutor.execute(
    pendingQuestions,
    concurrency,
    checkpoint.runId,
    "evaluate",
    async ({ item: question, index, total }) => {
      const hypothesis = checkpoint.questions[question.questionId].phases.answer.hypothesis!
      const priorLlmCalls = cliCallsFromPhase(
        checkpoint.questions[question.questionId].phases.evaluate
      )
      let llmCall: CliCallTelemetry | undefined

      const startTime = Date.now()
      checkpointManager.updatePhase(checkpoint, question.questionId, "evaluate", {
        status: "in_progress",
        startedAt: new Date().toISOString(),
      })

      try {
        const searchResults = checkpoint.questions[question.questionId].phases.search.results || []

        const evaluation = judge.evaluate({
          question: question.question,
          questionType: question.questionType,
          groundTruth: question.groundTruth,
          hypothesis,
          providerPrompts: provider?.prompts,
        })
        const [result, retrievalMetrics] = cliLlmBackend()
          ? [await evaluation, undefined]
          : await Promise.all([
              evaluation,
              calculateRetrievalMetrics(
                judge.getModel(),
                question.question,
                question.groundTruth,
                searchResults
              ),
            ])
        llmCall = result.execution

        const durationMs = Date.now() - startTime
        checkpointManager.updatePhase(checkpoint, question.questionId, "evaluate", {
          status: "completed",
          score: result.score,
          label: result.label,
          explanation: result.explanation,
          llmCall,
          llmCalls: llmCall ? [...priorLlmCalls, llmCall] : priorLlmCalls,
          retrievalMetrics,
          completedAt: new Date().toISOString(),
          durationMs,
        })

        const retrievalInfo = retrievalMetrics
          ? ` | Hit@${retrievalMetrics.k}=${retrievalMetrics.hitAtK}, MRR=${retrievalMetrics.mrr.toFixed(2)}`
          : ""
        logger.progress(
          index + 1,
          total,
          `Evaluated ${question.questionId}: ${result.label}${retrievalInfo} (${durationMs}ms)`
        )

        return { questionId: question.questionId, durationMs, label: result.label }
      } catch (e) {
        llmCall ||= cliCallTelemetryFromError(e)
        const error = e instanceof Error ? e.message : String(e)
        checkpointManager.updatePhase(checkpoint, question.questionId, "evaluate", {
          status: "failed",
          error,
          llmCall,
          llmCalls: llmCall ? [...priorLlmCalls, llmCall] : priorLlmCalls,
        })
        logger.error(`Failed to evaluate ${question.questionId}: ${error}`)
        throw new Error(
          `Evaluate failed at ${question.questionId}: ${error}. Fix the issue and resume with the same run ID.`
        )
      }
    }
  )

  updateJudgeCliLedger(checkpoint, checkpointManager)

  logger.success("Evaluate phase complete")
}

function updateJudgeCliLedger(
  checkpoint: RunCheckpoint,
  checkpointManager: CheckpointManager
): void {
  if (!checkpoint.judgeProvenance) return
  const phases = Object.values(checkpoint.questions).map((question) => question.phases.evaluate)
  const ledger = summarizeCliLedger(phases)
  if (ledger.callCount === 0) return
  Object.assign(checkpoint.judgeProvenance, {
    callCount: ledger.callCount,
    retryCount: ledger.retryCount,
    executionIdentityCount: ledger.executionIdentityCount,
    mixedExecutionIdentity: ledger.mixedExecutionIdentity,
    callLedgerComplete: ledger.callLedgerComplete,
  })
  reconcileCliProvenanceIdentity(checkpoint.judgeProvenance, ledger.calls)
  checkpointManager.save(checkpoint)
}
