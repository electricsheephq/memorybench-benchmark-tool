import { exactEvidenceReference } from "../prompts/evidence-cards"

export type DeterministicOperation =
  | "count"
  | "sum"
  | "difference"
  | "date_diff_days"
  | "order"
  | "ordinal"

export interface TypedOperand {
  evidenceIndex: number
  exactRef: string
  quote: string
  key?: string
  value?: number
  unit?: string
  date?: string
  label?: string
  source?: "evidence_card" | "question_date"
}

export interface DeterministicOperationRequest {
  operation: DeterministicOperation
  operands: TypedOperand[]
  direction?: "first_minus_second" | "second_minus_first" | "absolute"
}

export interface DeterministicOperationTrace {
  operation: DeterministicOperation
  result: string
  resultValue: number | string[]
  operandProvenance: Array<{ evidenceIndex: number; exactRef: string; quote: string }>
}

export type DeterministicOperationDecision =
  | { status: "computed"; trace: DeterministicOperationTrace }
  | { status: "fallback"; reason: string }

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function normalizedNumber(value: number): string {
  return String(value).replace(/\.0+$/, "")
}

function normalizeExactMatchText(value: string): string {
  return value
    .normalize("NFKC")
    .replace(/[\u2018\u2019\u201A\u201B\u2032\u2035]/g, "'")
    .replace(/[\u201C\u201D\u201E\u201F\u2033\u2036]/g, '"')
    .replace(/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]/g, "-")
}

function valueAppearsInQuote(value: number, quote: string): boolean {
  const escaped = normalizedNumber(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  return new RegExp(`(^|[^\\d])${escaped.replace(/\\\./g, "[.,]")}(?=$|[^\\d])`).test(
    quote.replace(/,/g, "")
  )
}

function utcDay(value: string): number | undefined {
  if (!/^\d{4}-\d{2}-\d{2}/.test(value)) return undefined
  const date = new Date(`${value.slice(0, 10)}T00:00:00.000Z`)
  return Number.isNaN(date.getTime()) ? undefined : date.getTime() / 86_400_000
}

function cite(
  context: unknown[],
  operand: TypedOperand,
  questionDate?: string
): { exactRef: string } | { reason: string } {
  if (operand.source === "question_date") {
    const canonicalQuestionDate = questionDate?.replace(/\//g, "-").slice(0, 10)
    if (
      operand.evidenceIndex !== -1 ||
      operand.exactRef !== "question-date" ||
      !questionDate ||
      operand.quote !== questionDate ||
      operand.date !== canonicalQuestionDate
    )
      return { reason: "question-date operand is not an exact canonical question date" }
    return { exactRef: "question-date" }
  }
  const item = context[operand.evidenceIndex]
  if (!record(item) || typeof item.content !== "string" || !record(item.metadata))
    return { reason: "operand evidence index is not a card item" }
  if (
    !operand.quote ||
    !normalizeExactMatchText(item.content).includes(normalizeExactMatchText(operand.quote))
  )
    return { reason: "operand quote is not an exact source substring" }
  let exactRef: string
  try {
    exactRef = exactEvidenceReference(item.content, item.metadata)
  } catch {
    return { reason: "operand source has no validated exact reference" }
  }
  if (operand.exactRef !== exactRef)
    return { reason: "operand exact reference does not match cited evidence" }
  return { exactRef }
}

function validateNumericOperand(
  context: unknown[],
  operand: TypedOperand,
  questionDate?: string
): { exactRef: string } | { reason: string } {
  const citation = cite(context, operand, questionDate)
  if ("reason" in citation) return citation
  if (!Number.isFinite(operand.value))
    return { reason: "numeric operand is missing a finite value" }
  if (!valueAppearsInQuote(operand.value!, operand.quote))
    return { reason: "numeric operand value is not explicit in its cited quote" }
  return citation
}

export function inferOperation(question: string): DeterministicOperation | undefined {
  const normalized = question.toLowerCase()
  if (/how many days|days ago|days between|date difference/.test(normalized))
    return "date_diff_days"
  if (/chronolog|in order|order of/.test(normalized)) return "order"
  if (
    /difference|compar(?:e|ed|ing)\s+(?:to|with)|how (?:much|many).*?(?:save|more|less|earlier|later)|minus|remain|left/.test(
      normalized
    )
  )
    return "difference"
  if (/\b(first|second|third|fourth)\b/.test(normalized)) return "ordinal"
  if (/total|combined|altogether|sum|add/.test(normalized)) return "sum"
  if (/how many|number of|count/.test(normalized)) return "count"
  return undefined
}

/**
 * Engagement predicate for the deterministic-operation layer (C2).
 *
 * C2 is part of the evidence-card arm: in card mode it engages by default on
 * every question whose phrasing requires a supported operation (count, sum,
 * difference, date arithmetic, ordering, ordinal selection). It previously
 * required an explicit HERMES_MB_DETERMINISTIC_OPERATIONS=1 opt-in that the
 * clone-and-resume launch paths never set, so real card-mode runs recorded
 * `not_attempted` on the exact count/date questions C2 was built for
 * (V1L1-LOSS8 diagnostic, 6/6 card-caused losses).
 *
 * Env override semantics:
 *   unset  -> attempt in card mode when the question requires an operation
 *   "1"    -> attempt on every card-mode question (force)
 *   "0"    -> never attempt (kill switch)
 * Never engages outside evidence-card mode: operand citation validates exact
 * card references, which only the card renderer produces.
 */
export function shouldAttemptDeterministicOperation(
  question: string,
  presentationMode: string,
  envSetting: string | undefined = process.env.HERMES_MB_DETERMINISTIC_OPERATIONS
): boolean {
  if (presentationMode !== "evidence_cards_v1") return false
  if (envSetting === "0") return false
  if (envSetting === "1") return true
  return inferOperation(question) !== undefined
}

export function parseDeterministicOperationRequest(
  raw: string
): DeterministicOperationRequest | undefined {
  const fenced = raw
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim()
  const first = fenced.indexOf("{")
  const last = fenced.lastIndexOf("}")
  if (first < 0 || last <= first) return undefined
  try {
    const candidate: unknown = JSON.parse(fenced.slice(first, last + 1))
    if (
      !record(candidate) ||
      !["count", "sum", "difference", "date_diff_days", "order", "ordinal"].includes(
        candidate.operation as string
      ) ||
      !Array.isArray(candidate.operands)
    )
      return undefined
    const operands: TypedOperand[] = []
    for (const value of candidate.operands) {
      if (
        !record(value) ||
        !Number.isInteger(value.evidenceIndex) ||
        typeof value.exactRef !== "string" ||
        typeof value.quote !== "string"
      )
        return undefined
      if (
        value.value !== undefined &&
        (typeof value.value !== "number" || !Number.isFinite(value.value))
      )
        return undefined
      if (value.date !== undefined && typeof value.date !== "string") return undefined
      operands.push({
        evidenceIndex: value.evidenceIndex as number,
        exactRef: value.exactRef,
        quote: value.quote,
        key: typeof value.key === "string" ? value.key : undefined,
        value: value.value as number | undefined,
        unit: typeof value.unit === "string" ? value.unit : undefined,
        date: value.date as string | undefined,
        label: typeof value.label === "string" ? value.label : undefined,
        source:
          value.source === "question_date"
            ? "question_date"
            : value.source === undefined || value.source === "evidence_card"
              ? "evidence_card"
              : undefined,
      })
    }
    return {
      operation: candidate.operation as DeterministicOperation,
      operands,
      direction:
        candidate.direction === "first_minus_second" ||
        candidate.direction === "second_minus_first" ||
        candidate.direction === "absolute"
          ? candidate.direction
          : undefined,
    }
  } catch {
    return undefined
  }
}

export function validateDeterministicOperation(
  request: DeterministicOperationRequest,
  context: unknown[],
  question: string,
  questionDate?: string
): DeterministicOperationDecision {
  const required = inferOperation(question)
  if (required && required !== request.operation)
    return {
      status: "fallback",
      reason: `operation ${request.operation} conflicts with question-derived ${required}`,
    }
  if (request.operands.length === 0) return { status: "fallback", reason: "no cited operands" }
  const provenance: DeterministicOperationTrace["operandProvenance"] = []
  for (const operand of request.operands) {
    const citation = cite(context, operand, questionDate)
    if ("reason" in citation) return { status: "fallback", reason: citation.reason }
    provenance.push({
      evidenceIndex: operand.evidenceIndex,
      exactRef: citation.exactRef,
      quote: operand.quote,
    })
  }
  if (request.operation === "count") {
    if (request.operands.some((operand) => !operand.key))
      return { status: "fallback", reason: "count operand is missing a canonical key" }
    const countValues = request.operands.map((operand) => operand.value)
    if (countValues.some((value) => value !== undefined)) {
      if (countValues.some((value) => value === undefined))
        return {
          status: "fallback",
          reason: "count operands must use either keys or explicit quantities consistently",
        }
      for (const operand of request.operands) {
        const verified = validateNumericOperand(context, operand, questionDate)
        const singularOne = operand.value === 1 && /\b(?:a|an|one)\b/i.test(operand.quote)
        if ("reason" in verified && !singularOne)
          return { status: "fallback", reason: verified.reason }
      }
    }
    const resultValue = countValues.every((value) => value !== undefined)
      ? countValues.reduce((total, value) => total + value!, 0)
      : new Set(request.operands.map((operand) => operand.key)).size
    return {
      status: "computed",
      trace: {
        operation: request.operation,
        result: String(resultValue),
        resultValue,
        operandProvenance: provenance,
      },
    }
  }
  if (request.operation === "date_diff_days") {
    if (request.operands.length !== 2)
      return { status: "fallback", reason: "date difference requires exactly two operands" }
    const days = request.operands.map((operand) => operand.date && utcDay(operand.date))
    if (days.some((day) => day === undefined))
      return { status: "fallback", reason: "date operand is missing a canonical date" }
    const resultValue = Math.abs(Number(days[1]) - Number(days[0]))
    return {
      status: "computed",
      trace: {
        operation: request.operation,
        result: `${resultValue} days`,
        resultValue,
        operandProvenance: provenance,
      },
    }
  }
  if (request.operation === "order") {
    const dated = request.operands.map((operand) => ({
      operand,
      day: operand.date && utcDay(operand.date),
    }))
    if (dated.some((item) => item.day === undefined || !item.operand.label))
      return { status: "fallback", reason: "ordered operands require canonical dates and labels" }
    const resultValue = dated
      .sort((a, b) => Number(a.day) - Number(b.day))
      .map((item) => item.operand.label!)
    return {
      status: "computed",
      trace: {
        operation: request.operation,
        result: resultValue.join(", "),
        resultValue,
        operandProvenance: provenance,
      },
    }
  }
  if (request.operation === "ordinal") {
    if (request.operands.length !== 1 || !request.operands[0].label)
      return { status: "fallback", reason: "ordinal selection requires one exact labeled operand" }
    return {
      status: "computed",
      trace: {
        operation: request.operation,
        result: request.operands[0].label,
        resultValue: [request.operands[0].label],
        operandProvenance: provenance,
      },
    }
  }
  const verified = request.operands.map((operand) =>
    validateNumericOperand(context, operand, questionDate)
  )
  const invalid = verified.find((result) => "reason" in result)
  if (invalid && "reason" in invalid) return { status: "fallback", reason: invalid.reason }
  const units = new Set(
    request.operands.map((operand) => operand.unit?.trim().toLowerCase()).filter(Boolean)
  )
  if (units.size > 1)
    return { status: "fallback", reason: "numeric operands have incompatible units" }
  if (request.operation === "sum") {
    const resultValue = request.operands.reduce((total, operand) => total + operand.value!, 0)
    return {
      status: "computed",
      trace: {
        operation: request.operation,
        result: `${resultValue}${units.size ? ` ${[...units][0]}` : ""}`,
        resultValue,
        operandProvenance: provenance,
      },
    }
  }
  if (request.operands.length !== 2)
    return { status: "fallback", reason: "difference requires exactly two operands" }
  const [first, second] = request.operands.map((operand) => operand.value!)
  const resultValue =
    request.direction === "second_minus_first"
      ? second - first
      : request.direction === "absolute"
        ? Math.abs(first - second)
        : first - second
  return {
    status: "computed",
    trace: {
      operation: request.operation,
      result: `${resultValue}${units.size ? ` ${[...units][0]}` : ""}`,
      resultValue,
      operandProvenance: provenance,
    },
  }
}

export function buildDeterministicOperationSelectorPrompt(
  question: string,
  cards: string,
  questionDate?: string
): string {
  return `Select typed operands from evidence cards; do not answer. Return one JSON object only.\nQuestion: ${question}\nQuestion Date: ${questionDate || "Not specified"}\nCards:\n${cards}\n\nSchema: {"operation":"count|sum|difference|date_diff_days|order|ordinal","direction":"first_minus_second|second_minus_first|absolute","operands":[{"evidenceIndex":0,"exactRef":"lcm:store:start-end","quote":"exact substring","key":"required for count","label":"required for order or ordinal","value":1,"unit":"optional","date":"YYYY-MM-DD","source":"evidence_card"}]}\nThe Question Date may be an operand only as {"evidenceIndex":-1,"exactRef":"question-date","quote":"exact Question Date text","date":"YYYY-MM-DD","source":"question_date"}. Every other operand must quote an exact card substring and exact card reference. If an operand is uncertain, return no operands.`
}

export function appendDeterministicTrace(
  prompt: string,
  trace: DeterministicOperationTrace
): string {
  const citations = trace.operandProvenance.map((operand) => operand.exactRef).join(", ")
  return prompt.replace(
    /\nAnswer:\s*$/,
    `\n\nValidated deterministic operation (${trace.operation}) from ${citations}: ${trace.result}. Use this result only for the supported computed claim.\n\nAnswer:`
  )
}
