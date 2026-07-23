import { Database } from "bun:sqlite"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { exactEvidenceReference } from "../src/prompts/evidence-cards"
import {
  validateDeterministicOperation,
  type DeterministicOperationRequest,
  type TypedOperand,
} from "../src/orchestrator/deterministic-operations"

const frozenRoot =
  process.env.V1L1_FROZEN_ROOT ||
  "/Volumes/LEXAR/Codex/session-notes/2026-07-21/hermes-lcm-autonomous-release-program/artifacts/m450/workdir"

type Card = { content: string; metadata: Record<string, unknown> }
function dbPath(questionId: string) {
  return join(frozenRoot, `${questionId}-hermes-lcm-voyage-rich-500q.db`)
}
function card(questionId: string, storeId: number): Card {
  const db = new Database(dbPath(questionId), { readonly: true })
  const row = db
    .query("select session_id, role, content, timestamp from messages where store_id = ?")
    .get(storeId) as { session_id: string; role: string; content: string; timestamp: number } | null
  db.close()
  if (!row) throw new Error(`${questionId} store ${storeId} not found in frozen evidence`)
  const dates = JSON.parse(
    readFileSync(dbPath(questionId).replace(/\.db$/, ".dates.json"), "utf8")
  ) as Record<string, string>
  return {
    content: row.content,
    metadata: {
      session_id: row.session_id,
      date: dates[row.session_id] || new Date(row.timestamp * 1000).toISOString(),
      role: row.role,
      store_id: storeId,
      content_offset: 0,
      content_returned_chars: Array.from(row.content).length,
    },
  }
}
function operand(
  cards: Card[],
  cardIndex: number,
  needle: string,
  extra: Omit<TypedOperand, "evidenceIndex" | "exactRef" | "quote"> = {}
): TypedOperand {
  const item = cards[cardIndex]
  if (!item.content.includes(needle)) throw new Error(`frozen quote not found: ${needle}`)
  return {
    evidenceIndex: cardIndex,
    exactRef: exactEvidenceReference(item.content, item.metadata),
    quote: needle,
    ...extra,
  }
}
function run(
  id: string,
  question: string,
  cards: Card[],
  request: DeterministicOperationRequest,
  expected?: number | string[],
  questionDate?: string
) {
  const decision = validateDeterministicOperation(request, cards, question, questionDate)
  const correct =
    decision.status === "computed" &&
    (expected === undefined ||
      JSON.stringify(decision.trace.resultValue) === JSON.stringify(expected))
  return {
    id,
    correct,
    status: decision.status,
    result: decision.status === "computed" ? decision.trace.result : undefined,
    reason: decision.status === "fallback" ? decision.reason : undefined,
    refs: request.operands.map((item) => item.exactRef),
  }
}

const c7 = [card("gpt4_468eb063", 119)]
const c9 = [card("9ee3ecd6", 77), card("9ee3ecd6", 437)]
const c17 = [62, 66, 292, 322].map((store) => card("gpt4_a56e767c", store))
const c22 = [62, 68, 142, 190, 303, 425].map((store) => card("gpt4_7abb270c", store))
const c24 = [card("eaca4986", 431)]
const c27 = [83, 139, 342, 400].map((store) => card("d851d5ba", store))
const c29 = [card("09ba9854", 296), card("09ba9854", 138)]
const c41 = [card("d6062bb9", 150), card("d6062bb9", 202)]
const c45 = [card("8e91e7d9", 198), card("8e91e7d9", 226)]
const c46 = [card("a11281a2", 219), card("a11281a2", 345)]
const c49 = [33, 107, 281, 337, 446].map((store) => card("gpt4_ab202e7f", store))

const cases = [
  run(
    "gpt4_468eb063",
    "How many days ago did I meet Emma?",
    c7,
    {
      operation: "date_diff_days",
      operands: [
        operand(c7, 0, "I catch up with Emma", { date: "2023-04-11" }),
        {
          evidenceIndex: -1,
          exactRef: "question-date",
          quote: "2023/04/20 (Thu) 10:12",
          date: "2023-04-20",
          source: "question_date",
        },
      ],
    },
    9,
    "2023/04/20 (Thu) 10:12"
  ),
  run(
    "9ee3ecd6",
    "How many more points do I need?",
    c9,
    {
      operation: "difference",
      direction: "first_minus_second",
      operands: [
        operand(c9, 1, "300 points", { value: 300, unit: "points" }),
        operand(c9, 0, "200 points", { value: 200, unit: "points" }),
      ],
    },
    100
  ),
  run(
    "gpt4_a56e767c",
    "How many film festivals did I attend?",
    c17,
    {
      operation: "count",
      operands: [
        operand(c17, 0, "Austin Film Festival", { key: "austin" }),
        operand(c17, 1, "Seattle International Film Festival", { key: "seattle" }),
        operand(c17, 2, "Portland Film Festival", { key: "portland" }),
        operand(c17, 3, "AFI Fest", { key: "afi" }),
      ],
    },
    4
  ),
  run("gpt4_7abb270c", "What is the chronological order of the six events?", c22, {
    operation: "order",
    operands: c22.map((_, index) =>
      operand(c22, index, c22[index].content.slice(0, 24), {
        date: String(c22[index].metadata.date).slice(0, 10),
        label: `event-${index + 1}`,
      })
    ),
  }),
  run("eaca4986", "Which song was second?", c24, {
    operation: "ordinal",
    operands: [operand(c24, 0, "C D E F G A B A G F E D C", { label: "second-song" })],
  }),
  run(
    "d851d5ba",
    "What was the total raised by March 20?",
    c27,
    {
      operation: "sum",
      operands: [
        operand(c27, 0, "$1,000", { value: 1000, unit: "usd" }),
        operand(c27, 1, "$250", { value: 250, unit: "usd" }),
        operand(c27, 2, "$500", { value: 500, unit: "usd" }),
        operand(c27, 3, "$2,000", { value: 2000, unit: "usd" }),
      ],
    },
    3750
  ),
  run(
    "09ba9854",
    "How much would I save by taking the train?",
    c29,
    {
      operation: "difference",
      direction: "first_minus_second",
      operands: [
        operand(c29, 0, "$60", { value: 60, unit: "usd" }),
        operand(c29, 1, "$10", { value: 10, unit: "usd" }),
      ],
    },
    50
  ),
  run(
    "d6062bb9",
    "What is the combined view count?",
    c41,
    {
      operation: "sum",
      operands: [
        operand(c41, 0, "1,456", { value: 1456, unit: "views" }),
        operand(c41, 1, "542", { value: 542, unit: "views" }),
      ],
    },
    1998
  ),
  run(
    "8e91e7d9",
    "How many siblings do I have?",
    c45,
    {
      operation: "count",
      operands: [
        operand(c45, 0, "3 sisters", { key: "sisters", value: 3 }),
        operand(c45, 1, "a brother", { key: "brother", value: 1 }),
      ],
    },
    4
  ),
  run(
    "a11281a2",
    "What was the follower increase?",
    c46,
    {
      operation: "difference",
      direction: "first_minus_second",
      operands: [
        operand(c46, 0, "350 followers", { value: 350, unit: "followers" }),
        operand(c46, 1, "250 followers", { value: 250, unit: "followers" }),
      ],
    },
    100
  ),
  run(
    "gpt4_ab202e7f",
    "How many kitchen items did I replace or fix?",
    c49,
    {
      operation: "count",
      operands: [
        operand(c49, 0, "fixed the kitchen shelves", { key: "shelves" }),
        operand(c49, 1, "replaced the worn-out kitchen mat", { key: "mat" }),
        operand(c49, 2, "replaced it with a toaster oven", { key: "toaster" }),
        operand(c49, 3, "replaced my old kitchen faucet", { key: "faucet" }),
        operand(c49, 4, "old coffee maker", { key: "coffee-maker" }),
      ],
    },
    5
  ),
  {
    id: "2b8f3739",
    correct: false,
    status: "fallback",
    reason:
      "factorized price requires a multiplicative operand; C2 intentionally has no fabricated derived operand",
    refs: [],
  },
  {
    id: "92a0aa75",
    correct: false,
    status: "fallback",
    reason: "mixed year-month duration subtraction is outside C2 canonical numeric operands",
    refs: [],
  },
]

const output = {
  suite: "V1-L1 frozen deterministic-operation replay",
  frozenRoot,
  extractorCorrect: cases.filter((item) => item.correct).length,
  total: cases.length,
  threshold: 11,
  passed: cases.filter((item) => item.correct).length >= 11,
  cases,
}
console.log(JSON.stringify(output, null, 2))
if (!output.passed) process.exitCode = 1
