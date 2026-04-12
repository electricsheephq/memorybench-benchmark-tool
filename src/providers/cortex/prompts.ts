import type { ProviderPrompts } from "../../types/prompts"

interface CortexClaimMetadata {
  event_date?: string
  date?: string
  timestamp?: string
  session_date?: string
  speaker?: string
  source_session_id?: string
}

interface CortexClaim {
  id?: string
  memory?: string
  fact?: string
  subject?: string
  predicate?: string
  object?: string
  content?: string
  text?: string
  metadata?: CortexClaimMetadata
}

function isEnumerationQuestion(question: string): boolean {
  const normalized = question.toLowerCase()
  return [
    "all ",
    "list",
    "which ",
    "what were the different",
    "name every",
    "name all",
    "what are the",
  ].some((marker) => normalized.includes(marker))
}

function pickClaimDate(claim: CortexClaim): string | undefined {
  return (
    claim.metadata?.event_date ||
    claim.metadata?.date ||
    claim.metadata?.timestamp ||
    claim.metadata?.session_date
  )
}

function stringifyValue(value: unknown): string {
  if (value === null || value === undefined) return ""
  if (typeof value === "string") return value.trim()
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (Array.isArray(value))
    return value
      .map((item) => stringifyValue(item))
      .filter(Boolean)
      .join(", ")
  if (typeof value === "object") return JSON.stringify(value)
  return String(value)
}

function formatClaim(claim: CortexClaim, index: number): string {
  const date = pickClaimDate(claim)
  const datePrefix = date ? ` [${date}]` : ""

  const subject = stringifyValue(claim.subject)
  const predicate = stringifyValue(claim.predicate)
  const object = stringifyValue(claim.object)

  if (subject || predicate || object) {
    const speaker = stringifyValue(claim.metadata?.speaker)
    const sourceSession = stringifyValue(claim.metadata?.source_session_id)
    const extras = [
      speaker ? `speaker=${speaker}` : "",
      sourceSession ? `session=${sourceSession}` : "",
    ]
      .filter(Boolean)
      .join(" | ")
    const extraSuffix = extras ? ` (${extras})` : ""
    return `[${index}]${datePrefix} ${subject || "?"} | ${predicate || "?"} | ${object || "?"}${extraSuffix}`
  }

  const fallback =
    stringifyValue(claim.memory) ||
    stringifyValue(claim.fact) ||
    stringifyValue(claim.content) ||
    stringifyValue(claim.text) ||
    JSON.stringify(claim)

  return `[${index}]${datePrefix} ${fallback}`
}

function buildCortexEvidenceList(context: unknown[]): string {
  const claims = context as CortexClaim[]

  if (claims.length === 0) {
    return "[no memories retrieved]"
  }

  return claims.map((claim, index) => formatClaim(claim, index + 1)).join("\n")
}

export function buildCortexAnswerPrompt(
  question: string,
  context: unknown[],
  questionDate?: string
): string {
  const evidenceList = buildCortexEvidenceList(context)
  const enumerationInstructions = isEnumerationQuestion(question)
    ? `
Enumeration guidance:
- This question may require multiple items, not just one.
- Scan all evidence lines before answering.
- Include every distinct item directly supported by the evidence.
- Do not stop after the first matching memory.
- If the evidence only supports a partial list, say that explicitly instead of inventing missing items.`
    : ""

  return `You are answering a benchmark question using retrieved Cortex memories.

Question: ${question}
Question Date: ${questionDate || "Not specified"}

Retrieved Memories:
${evidenceList}

Instructions:
- Treat each memory line as evidence. Base your answer only on the retrieved memories above.
- Each memory is formatted as: [number] [date?] subject | predicate | object.
- Pay close attention to dates and event ordering.
- When memories contain explicit event dates, prioritize those dates over relative phrases.
- Resolve relative time references (for example: yesterday, last week, next month) using the evidence dates and question date when possible.
- If memories conflict, prefer the most recent directly relevant evidence.
- Answer directly and concisely.
- If you cannot find direct evidence for the answer, you MUST respond exactly with: I don't know
- Do NOT infer, guess, or fill gaps from world knowledge or plausibility.
- Do not use information that is not supported by the retrieved memories.${enumerationInstructions}

Answer format:
- Return only the answer.
- Use specific dates or names when the evidence supports them.
- For list questions, provide the full evidence-supported list in a compact form.`
}

export const CORTEX_PROMPTS: ProviderPrompts = {
  answerPrompt: buildCortexAnswerPrompt,
}

export default CORTEX_PROMPTS
