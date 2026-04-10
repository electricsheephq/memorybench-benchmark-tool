import type { ProviderPrompts } from "../../types/prompts"

interface CortexMemoryItem {
  id?: string
  content?: string
  score?: number
  category?: string
  entity?: string
  predicate?: string
  temporal_marker?: string
  created_at?: string
  event_date?: string
  metadata?: Record<string, unknown>
}

export function buildCortexAnswerPrompt(
  question: string,
  context: unknown[],
  questionDate?: string
): string {
  const claimsStr = context
    .map((r, i) => {
      const item = r as CortexMemoryItem
      const parts: string[] = []

      // Temporal context
      const date = item.event_date || item.created_at
      if (date) parts.push(`[Date: ${typeof date === "string" ? date.split("T")[0] : date}]`)
      if (item.temporal_marker && item.temporal_marker !== "current")
        parts.push(`[${item.temporal_marker}]`)

      // Core claim content
      const content = item.content || ""
      if (item.entity && content && !content.toLowerCase().startsWith(item.entity.toLowerCase())) {
        parts.push(`${item.entity}: ${content}`)
      } else {
        parts.push(content)
      }

      // Category tag
      if (item.category) parts.push(`(${item.category})`)

      // Relevance score
      if (item.score != null) parts.push(`[relevance: ${item.score.toFixed(2)}]`)

      return `[${i + 1}] ${parts.join(" ")}`
    })
    .join("\n\n")

  const dateContext = questionDate
    ? `\nThe question is being asked as of: ${questionDate}\n`
    : ""

  return `You are an intelligent memory assistant. You have access to structured memory claims extracted from past conversations. Each claim represents a specific fact, preference, event, or relationship.

Key instructions:
- Claims are ordered by relevance score — higher-ranked claims are more likely to answer the question
- Pay attention to temporal markers (current, past, future) and dates — facts can change over time
- When claims contradict, prefer the most recent one (by date) or the one marked "current"
- Claims tagged with a person's name (entity) describe that specific person
- Look for direct evidence before making inferences
- If no claims are relevant, say you don't have enough information
${dateContext}
Retrieved Memory Claims:
${claimsStr}

Question: ${question}

Answer concisely and directly based on the memory claims above.`
}

/**
 * Cortex provider prompts configuration.
 */
export const CORTEX_PROMPTS: ProviderPrompts = {
  answerPrompt: buildCortexAnswerPrompt,
}

export default CORTEX_PROMPTS
