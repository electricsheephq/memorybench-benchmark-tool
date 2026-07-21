import { afterEach, describe, expect, test } from "bun:test"
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { basename, join } from "node:path"
import { tmpdir } from "node:os"
import type { Benchmark } from "../types/benchmark"
import type { CliCallTelemetry } from "../utils/cli-llm"
import { CheckpointManager } from "./checkpoint"
import { buildAnswerPrompt, runAnswerPhase } from "./phases/answer"
import { generateReport } from "./phases/report"
import { questionCheckpointMetadata, syncQuestionCheckpointMetadata } from "./question-metadata"

const tempPaths: string[] = []

function tempDir(prefix: string): string {
  const path = mkdtempSync(join(tmpdir(), prefix))
  tempPaths.push(path)
  return path
}

afterEach(() => {
  for (const path of tempPaths.splice(0)) rmSync(path, { recursive: true, force: true })
})

describe("question-date checkpoint contract", () => {
  test("initialization, resume, prompt, copy, and legacy backfill preserve questionDate", async () => {
    const root = tempDir("memorybench-v1-date-")
    const manager = new CheckpointManager(root)
    const checkpoint = manager.create("source", "rag", "longmemeval", "gpt-4o", "gpt-4o")
    const question = {
      questionId: "fixture-q",
      question: "What happened five days ago?",
      groundTruth: "The event",
      questionType: "temporal-reasoning",
      haystackSessionIds: [],
      metadata: { questionDate: "2023/03/20" },
    }

    manager.initQuestion(
      checkpoint,
      question.questionId,
      "fixture-container",
      questionCheckpointMetadata(question)
    )
    const resultPath = join(manager.getResultsDir("source"), "fixture-q.json")
    writeFileSync(resultPath, JSON.stringify({ results: [] }))
    checkpoint.questions[question.questionId]!.phases.search = {
      status: "completed",
      resultFile: resultPath,
    }
    manager.save(checkpoint)
    await manager.flush("source")

    const resumed = manager.load("source")!
    expect(resumed.questions[question.questionId]?.questionDate).toBe("2023/03/20")
    expect(
      buildAnswerPrompt(question.question, [], resumed.questions[question.questionId]?.questionDate)
    ).toContain("Question Date: 2023/03/20")

    const copied = manager.copyCheckpoint("source", "copy", "answer")
    await manager.flush("copy")
    const copiedResult = copied.questions[question.questionId]?.phases.search.resultFile
    expect(copied.questions[question.questionId]?.questionDate).toBe("2023/03/20")
    expect(copiedResult).toBe(join(manager.getResultsDir("copy"), basename(resultPath)))
    manager.delete("source")
    expect(copiedResult && existsSync(copiedResult)).toBe(true)
    expect(manager.load("copy")?.questions[question.questionId]?.questionDate).toBe("2023/03/20")

    const legacy = manager.create("legacy", "rag", "longmemeval", "gpt-4o", "gpt-4o")
    manager.initQuestion(legacy, question.questionId, "legacy-container", {
      question: question.question,
      groundTruth: question.groundTruth,
      questionType: question.questionType,
    })
    expect(syncQuestionCheckpointMetadata(legacy, [question])).toBe(1)
    expect(legacy.questions[question.questionId]?.questionDate).toBe("2023/03/20")
  })

  test("copy and answer fail closed when completed search artifacts are missing", async () => {
    const root = tempDir("memorybench-v1-copy-failure-")
    const manager = new CheckpointManager(root)
    const checkpoint = manager.create("source", "rag", "longmemeval", "gpt-4o", "gpt-4o")
    const questions = [
      {
        questionId: "present-q",
        question: "Present?",
        groundTruth: "yes",
        questionType: "single-session-user",
        haystackSessionIds: [],
      },
      {
        questionId: "missing-q",
        question: "Missing?",
        groundTruth: "yes",
        questionType: "single-session-user",
        haystackSessionIds: [],
      },
    ]
    for (const question of questions) {
      manager.initQuestion(checkpoint, question.questionId, "fixture-container", {
        question: question.question,
        groundTruth: question.groundTruth,
        questionType: question.questionType,
      })
    }
    const presentPath = join(manager.getResultsDir("source"), "present-q.json")
    const missingPath = join(manager.getResultsDir("source"), "missing-q.json")
    writeFileSync(presentPath, JSON.stringify({ results: [] }))
    checkpoint.questions["present-q"]!.phases.search = {
      status: "completed",
      resultFile: presentPath,
    }
    checkpoint.questions["missing-q"]!.phases.search = {
      status: "completed",
      resultFile: missingPath,
    }
    manager.save(checkpoint)
    await manager.flush("source")

    let copyError: unknown
    try {
      manager.copyCheckpoint("source", "broken-copy", "answer")
    } catch (error) {
      copyError = error
    }
    expect(copyError).toBeInstanceOf(Error)
    expect(String(copyError)).toContain("completed search result")
    expect(existsSync(manager.getRunPath("broken-copy"))).toBe(false)

    const benchmark = {
      name: "longmemeval",
      load: async () => {},
      getQuestions: () => questions,
      getHaystackSessions: () => [],
      getGroundTruth: () => "yes",
      getQuestionTypes: () => ({}),
    } satisfies Benchmark
    let answerError: unknown
    try {
      await runAnswerPhase(benchmark, checkpoint, manager)
    } catch (error) {
      answerError = error
    }
    expect(answerError).toBeInstanceOf(Error)
    expect(String(answerError)).toContain("completed search result file is missing")
  })
})

function call(model: string, effort: string, transportVersion: string): CliCallTelemetry {
  return {
    version: "memorybench-cli-call-v1",
    role: effort === "low" ? "judge" : "answerer",
    transport: "codex-cli",
    transportVersion,
    requested: {
      model,
      modelExplicit: true,
      reasoningEffort: effort,
      provider: "openai",
      serviceTier: "priority",
      pinSource: "explicit-cli-argv",
    },
    eventModelField: "not-emitted-by-codex-jsonl",
    attempts: [],
    usage: {
      inputTokens: 10,
      cachedInputTokens: 2,
      outputTokens: 3,
      reasoningOutputTokens: 1,
    },
    usageComplete: true,
    totalDurationMs: 25,
    retryCount: 0,
  }
}

describe("stored execution provenance", () => {
  test("report identity comes from stored calls, not current environment labels", () => {
    const root = tempDir("memorybench-v1-report-")
    const manager = new CheckpointManager(root)
    const checkpoint = manager.create("report", "rag", "longmemeval", "gpt-4o", "gpt-4o")
    manager.initQuestion(checkpoint, "fixture-q", "fixture-container", {
      question: "Question",
      groundTruth: "Answer",
      questionType: "single-session-user",
    })
    checkpoint.answererProvenance = {
      transport: "codex-cli",
      model: "stale-current-answerer",
      modelExplicit: true,
      configuredModel: "gpt-4o",
    }
    checkpoint.judgeProvenance = {
      transport: "codex-cli",
      model: "stale-current-judge",
      modelExplicit: true,
      configuredModel: "gpt-4o",
    }
    checkpoint.questions["fixture-q"]!.phases.answer = {
      status: "completed",
      hypothesis: "Answer",
      promptTokens: 10,
      basePromptTokens: 5,
      contextTokens: 5,
      llmCalls: [call("actual-answerer", "medium", "codex 1.2.3")],
    }
    checkpoint.questions["fixture-q"]!.phases.evaluate = {
      status: "completed",
      label: "correct",
      score: 1,
      explanation: "yes",
      llmCalls: [call("actual-judge", "low", "codex 1.2.3")],
    }

    const benchmark = {
      name: "longmemeval",
      load: async () => {},
      getQuestions: () => [
        {
          questionId: "fixture-q",
          question: "Question",
          groundTruth: "Answer",
          questionType: "single-session-user",
          haystackSessionIds: [],
        },
      ],
      getHaystackSessions: () => [],
      getGroundTruth: () => "Answer",
      getQuestionTypes: () => ({}),
    } satisfies Benchmark

    const report = generateReport(benchmark, checkpoint)
    expect(report.answererProvenance).toMatchObject({
      model: "actual-answerer",
      reasoningEffort: "medium",
      transportVersion: "codex 1.2.3",
    })
    expect(report.judgeProvenance).toMatchObject({
      model: "actual-judge",
      reasoningEffort: "low",
      transportVersion: "codex 1.2.3",
    })
    expect(report.cliLedger).toMatchObject({
      answerer: { callCount: 1, callLedgerComplete: true },
      judge: { callCount: 1, callLedgerComplete: true },
    })
  })
})
