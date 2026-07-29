import { existsSync, mkdirSync, statSync, writeFileSync } from "node:fs"
import { join, resolve } from "node:path"
import { LoCoMoBenchmark, LOCOMO_QUESTION_TYPES } from "../src/benchmarks/locomo"
import { generateReport } from "../src/orchestrator/phases/report"
import { HermesLcmProvider } from "../src/providers/hermes-lcm"
import type { RunCheckpoint, QuestionCheckpoint } from "../src/types/checkpoint"

const DEFAULT_LOG_ROOT =
  "/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-r3-1/artifacts/laneLOCOMO-logs"
const DATASET_PATH = resolve("data/locomo10.json")
const QUESTION_TYPES = Object.keys(LOCOMO_QUESTION_TYPES)

function prepareOfflineEnvironment(workdir: string): void {
  process.env.HERMES_MB_PROVIDER = "fastembed"
  process.env.HERMES_MB_WORKDIR = workdir
  process.env.HF_HUB_OFFLINE = "1"
  process.env.TRANSFORMERS_OFFLINE = "1"
  delete process.env.HERMES_MB_LLM_CLI
  delete process.env.OPENAI_API_KEY
  delete process.env.ANTHROPIC_API_KEY
  delete process.env.GOOGLE_GENERATIVE_AI_API_KEY
  delete process.env.VOYAGE_API_KEY
}

function assertUnifiedSessions(
  questionId: string,
  sessions: ReturnType<LoCoMoBenchmark["getHaystackSessions"]>
): void {
  if (sessions.length === 0) throw new Error(`${questionId}: no UnifiedSession rows`)
  for (const session of sessions) {
    if (!session.sessionId || session.messages.length === 0) {
      throw new Error(`${questionId}: invalid UnifiedSession ${session.sessionId}`)
    }
    for (const message of session.messages) {
      if (!["user", "assistant"].includes(message.role) || !message.content || !message.speaker) {
        throw new Error(`${questionId}: invalid UnifiedMessage in ${session.sessionId}`)
      }
    }
  }
}

function checkpointFor(runId: string, questions: QuestionCheckpoint[], now: string): RunCheckpoint {
  return {
    runId,
    dataSourceRunId: runId,
    status: "running",
    provider: "hermes-lcm",
    benchmark: "locomo",
    judge: "not-called-offline",
    answeringModel: "not-called-offline",
    createdAt: now,
    updatedAt: now,
    targetQuestionIds: questions.map((question) => question.questionId),
    questions: Object.fromEntries(questions.map((question) => [question.questionId, question])),
  }
}

async function main(): Promise<void> {
  if (!existsSync(DATASET_PATH)) {
    throw new Error(`Pinned LoCoMo dataset is missing: ${DATASET_PATH}`)
  }

  const timestamp = new Date().toISOString().replaceAll(/[:.]/g, "-")
  const runId = `locomo-offline-${timestamp}`
  const outputRoot = resolve(process.env.HERMES_LOCOMO_OFFLINE_DIR || DEFAULT_LOG_ROOT)
  const runDir = join(outputRoot, runId)
  const storeDir = join(runDir, "stores")
  mkdirSync(storeDir, { recursive: true })
  prepareOfflineEnvironment(storeDir)

  const benchmark = new LoCoMoBenchmark()
  await benchmark.load({ dataPath: "data/locomo10.json" })

  const selected = QUESTION_TYPES.map((questionType) => {
    const question = benchmark.getQuestions({ questionTypes: [questionType], limit: 1 })[0]
    if (!question) throw new Error(`No LoCoMo question found for ${questionType}`)
    return question
  })

  const provider = new HermesLcmProvider()
  const checkpoints: QuestionCheckpoint[] = []
  const validationRows: Array<Record<string, unknown>> = []

  try {
    await provider.initialize({})
    for (const question of selected) {
      const sessions = benchmark.getHaystackSessions(question.questionId)
      assertUnifiedSessions(question.questionId, sessions)

      const containerTag = `${question.questionId}-${runId}`
      const ingestStarted = Date.now()
      const ingestResult = await provider.ingest(sessions, { containerTag })
      const ingestDurationMs = Date.now() - ingestStarted
      await provider.awaitIndexing(ingestResult, containerTag)

      const searchStarted = Date.now()
      const results = await provider.search(question.question, { containerTag, limit: 25 })
      const searchDurationMs = Date.now() - searchStarted
      if (results.length === 0) throw new Error(`${question.questionId}: search returned no hits`)

      const storePath = join(storeDir, `${containerTag.replaceAll(/[^A-Za-z0-9_.-]/g, "_")}.db`)
      if (!existsSync(storePath) || statSync(storePath).size === 0) {
        throw new Error(`${question.questionId}: Hermes-LCM store was not built`)
      }

      checkpoints.push({
        questionId: question.questionId,
        containerTag,
        question: question.question,
        groundTruth: question.groundTruth,
        questionType: question.questionType,
        sessions: sessions.map((session) => ({
          sessionId: session.sessionId,
          date: session.metadata?.date as string | undefined,
          messageCount: session.messages.length,
        })),
        phases: {
          ingest: {
            status: "completed",
            completedSessions: sessions.map((session) => session.sessionId),
            ingestResult,
            durationMs: ingestDurationMs,
          },
          indexing: {
            status: "completed",
            completedIds: ingestResult.documentIds,
            failedIds: [],
          },
          search: {
            status: "completed",
            results,
            resultCount: results.length,
            durationMs: searchDurationMs,
          },
          answer: { status: "pending" },
          evaluate: { status: "pending" },
        },
      })
      validationRows.push({
        questionId: question.questionId,
        questionType: question.questionType,
        sessionCount: sessions.length,
        messageCount: sessions.reduce((total, session) => total + session.messages.length, 0),
        documentCount: ingestResult.documentIds.length,
        searchHitCount: results.length,
        storePath,
        storeBytes: statSync(storePath).size,
      })
    }
  } finally {
    provider.close()
  }

  const now = new Date().toISOString()
  const scaffold = generateReport(benchmark, checkpointFor(runId, checkpoints, now))
  const report = {
    ...scaffold,
    offlineValidation: {
      status: "PASS",
      mode: "ingest-index-search-report-only",
      datasetPath: DATASET_PATH,
      selectedQuestionCount: selected.length,
      expectedQuestionTypes: QUESTION_TYPES,
      answererCalls: 0,
      judgeCalls: 0,
      storesBuilt: validationRows.length,
      questionsWithSearchHits: validationRows.filter((row) => Number(row.searchHitCount) > 0)
        .length,
      questions: validationRows,
    },
  }
  const reportPath = join(runDir, "report.json")
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`)
  const summary = {
    status: "PASS",
    reportPath,
    storeDir,
    selectedQuestionCount: selected.length,
    answererCalls: 0,
    judgeCalls: 0,
  }
  writeFileSync(join(runDir, "validation.log"), `${JSON.stringify(summary)}\n`)
  console.log(JSON.stringify(summary))
}

await main()
