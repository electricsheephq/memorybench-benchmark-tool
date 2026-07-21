import type { LanguageModel } from "ai"
import type { Judge, JudgeConfig, JudgeInput, JudgeResult } from "../types/judge"
import type { ProviderPrompts } from "../types/prompts"
import { buildJudgePrompt, parseJudgeResponse, getJudgePrompt } from "./base"
import { logger } from "../utils/logger"
import { CliCallError, cliComplete, cliLlmModelId, type CliCallTelemetry } from "../utils/cli-llm"

export class CliJudge implements Judge {
  name = "cli"

  async initialize(_config: JudgeConfig): Promise<void> {
    logger.info(`Initialized CLI judge (${cliLlmModelId("judge")})`)
  }

  async evaluate(input: JudgeInput): Promise<JudgeResult> {
    let execution: CliCallTelemetry | undefined
    try {
      const text = await cliComplete(buildJudgePrompt(input), {
        role: "judge",
        onTelemetry: (telemetry) => {
          execution = telemetry
        },
      })
      return { ...parseJudgeResponse(text), execution }
    } catch (error) {
      if (execution && !(error instanceof CliCallError)) {
        throw new CliCallError(error instanceof Error ? error.message : String(error), execution)
      }
      throw error
    }
  }

  getPromptForQuestionType(questionType: string, providerPrompts?: ProviderPrompts): string {
    return getJudgePrompt(questionType, providerPrompts)
  }

  getModel(): LanguageModel {
    throw new Error("CliJudge.getModel() is unavailable for CLI execution")
  }
}
