import { afterEach, expect, test } from "bun:test"
import { createHash } from "node:crypto"
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { join, relative, resolve } from "node:path"
import { tmpdir } from "node:os"
import { LoCoMoBenchmark } from "../benchmarks/locomo"
import { LongMemEvalBenchmark } from "../benchmarks/longmemeval"
import { ConvoMemBenchmark } from "../benchmarks/convomem"
import { emitIdentityFixture } from "./identity-fixture"
import { CheckpointManager } from "./checkpoint"
import { generateReport } from "./phases/report"
import { assertResumeIdentities, datasetIdentity, runIdentities } from "./identities"

const dirs: string[] = []
function temp() {
  const dir = mkdtempSync(join(tmpdir(), "identities-"))
  dirs.push(dir)
  return dir
}
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true })
})
const keys = [
  "datasetSha256",
  "datasetPath",
  "questionSetSha256",
  "readerPromptVersion",
  "promptVersion",
  "scoringPath",
  "answerPresentationMode",
] as const
const identity = () =>
  runIdentities(
    {
      name: "synthetic",
      ...datasetIdentity("./synthetic.json", Buffer.from("A")),
      getQuestions: () => [],
    } as any,
    { name: "fixture" },
    "judge"
  )

for (const [name, Loader, filename, content] of [
  ["locomo", LoCoMoBenchmark, "dataset.json", []],
  ["longmemeval", LongMemEvalBenchmark, "longmemeval_s_cleaned.json", []],
  ["convomem", ConvoMemBenchmark, "convomem_data.json", []],
] as const) {
  test(`${name}: hash loaded bytes, retain configured relative label, detect byte changes`, async () => {
    const dir = temp()
    const path = join(dir, filename)
    const label = relative(process.cwd(), name === "locomo" ? path : dir)
    writeFileSync(path, JSON.stringify(content))
    const first = new Loader()
    await first.load({ dataPath: label })
    expect(first.datasetPath).toBe(label)
    expect(first.datasetSha256).toBe(createHash("sha256").update(readFileSync(path)).digest("hex"))
    writeFileSync(path, JSON.stringify(content) + " ")
    const second = new Loader()
    await second.load({ dataPath: label })
    expect(second.datasetSha256).not.toBe(first.datasetSha256)
  })
}

test("LongMemEval hashes and consumes raw source even when split cache is stale", async () => {
  const dir = temp()
  mkdirSync(join(dir, "questions"))
  writeFileSync(join(dir, "questions", "stale.json"), "{}")
  writeFileSync(
    join(dir, "longmemeval_s_cleaned.json"),
    JSON.stringify([
      {
        question_id: "current",
        question: "Synthetic",
        question_type: "knowledge-update",
        answer: "Current",
        haystack_sessions: [],
        haystack_dates: [],
      },
    ])
  )
  const benchmark = new LongMemEvalBenchmark()
  await benchmark.load({ dataPath: relative(process.cwd(), dir) })
  expect(benchmark.getQuestions().map((q) => q.questionId)).toEqual(["current"])
})

test("first fetch bytes are hashed, and absolute dataset labels never serialize", async () => {
  const path = join(temp(), "fetched.json")
  const saved = globalThis.fetch
  globalThis.fetch = (async () => new Response("[] ")) as unknown as typeof fetch
  try {
    const benchmark = new LoCoMoBenchmark()
    await benchmark.load({ dataPath: relative(process.cwd(), path) })
    expect(benchmark.datasetSha256).toBe(createHash("sha256").update("[] ").digest("hex"))
    await expect(new LoCoMoBenchmark().load({ dataPath: path })).rejects.toThrow("datasetPath")
  } finally {
    globalThis.fetch = saved
  }
  for (const absolute of [
    "/private/dataset.json",
    "C:\\data\\dataset.json",
    "\\\\host\\dataset.json",
  ]) {
    expect(() => datasetIdentity(absolute, Buffer.from("[]"))).toThrow("datasetPath")
  }
})

test("fresh checkpoint and report carry all identities through the real writers", async () => {
  const root = temp()
  for (const [run, byte] of [
    ["a", "A"],
    ["b", "B"],
    ["c", "A"],
  ]) {
    await emitIdentityFixture(join(root, run!), byte)
  }
  const read = (run: string, file: string) =>
    JSON.parse(readFileSync(join(root, run, file), "utf8"))
  const a = read("a", "checkpoint.json")
  for (const file of ["checkpoint.json", "report.json"]) {
    const report = read("a", file)
    for (const key of keys) {
      expect(typeof report[key]).toBe("string")
      expect(report[key]).toBe(a[key])
      expect(read("c", file)[key]).toBe(a[key])
    }
    expect(read("b", file).datasetSha256).not.toBe(a.datasetSha256)
    for (const key of keys.filter((k) => k !== "datasetSha256"))
      expect(read("b", file)[key]).toBe(a[key])
  }
})

for (const mutation of ["none", "wrapper", "routing", "answer"] as const) {
  test(`rendered prompt identities track ${mutation} across processes`, () => {
    const result = Bun.spawnSync(
      [
        process.execPath,
        "-e",
        `
        import { mock } from "bun:test";
        const source = ${JSON.stringify(process.cwd())};
        const mutation = ${JSON.stringify(mutation)};
        if (mutation === "wrapper") {
          const base = await import(source + "/src/judges/base.ts");
          const build = base.buildJudgePrompt;
          mock.module(source + "/src/judges/base.ts", () => ({...base,
            buildJudgePrompt: input => build(input) + "changed wrapper"}));
        }
        if (mutation === "routing" || mutation === "answer") {
          const path = source + "/src/prompts/defaults.ts";
          const defaults = await import(path);
          const route = defaults.getJudgePromptForType;
          mock.module(path, () => ({...defaults,
            ...(mutation === "routing" ? {getJudgePromptForType: type =>
              type === "temporal" ? defaults.DEFAULT_JUDGE_PROMPT : route(type)} :
              {buildDefaultAnswerPrompt: () => "changed answer prompt"})}));
        }
        const { runIdentities } = await import(source + "/src/orchestrator/identities.ts");
        console.log(JSON.stringify(runIdentities({datasetPath:"./synthetic.json",
          datasetSha256:"unused",getQuestions:()=>[]}, {name:"fixture"}, "judge")));
      `,
      ],
      { cwd: process.cwd(), env: process.env, timeout: 10000 }
    )
    if (result.exitCode) throw new Error(result.stderr.toString())
    const actual = JSON.parse(result.stdout.toString())
    const expected = identity()
    expect(actual.promptVersion === expected.promptVersion).toBe(
      mutation !== "wrapper" && mutation !== "routing"
    )
    expect(actual.readerPromptVersion === expected.readerPromptVersion).toBe(mutation !== "answer")
  })
}

test("provider override source changes prompt identity and selects provider scoring path", () => {
  const benchmark = {
    ...datasetIdentity("data.json", Buffer.from("[]")),
    getQuestions: () => [],
  } as any
  const one = runIdentities(
    benchmark,
    { name: "custom", prompts: { judgePrompt: () => ({ default: "one" }) } },
    "judge"
  )
  const two = runIdentities(
    benchmark,
    { name: "custom", prompts: { judgePrompt: () => ({ default: "two" }) } },
    "judge"
  )
  expect(one.promptVersion).not.toBe(identity().promptVersion)
  expect(two.promptVersion).not.toBe(one.promptVersion)
  expect(one.scoringPath).toBe("judge/provider:custom/score-v1")
  expect(identity().scoringPath).toBe("judge/default/score-v1")
})

for (const key of [
  "datasetSha256",
  "questionSetSha256",
  "promptVersion",
  "readerPromptVersion",
  "scoringPath",
] as const) {
  test(`resume mismatch refuses and names ${key} before run work`, async () => {
    const root = temp()
    const manager = new CheckpointManager(root)
    const checkpoint = manager.create("resume", "fixture", "longmemeval", "judge", "reader", {
      identities: identity(),
    })
    await manager.flush()
    const loaded = manager.load("resume")!
    expect(() => assertResumeIdentities(loaded, { ...identity(), [key]: "changed" })).toThrow(key)
    expect(() => assertResumeIdentities(loaded, identity())).not.toThrow()
    expect(manager.load("resume")![key]).toBe(identity()[key])
  })
}

test("old checkpoint resumes without backfilling checkpoint or report, including copied runs", async () => {
  const manager = new CheckpointManager(temp())
  const checkpoint = manager.create("old", "fixture", "longmemeval", "judge", "reader")
  await manager.flush()
  const loaded = manager.load("old")!
  assertResumeIdentities(loaded, identity())
  manager.save(loaded)
  await manager.flush()
  const benchmark = { getQuestions: () => [], getQuestionTypes: () => ({}) } as any
  for (const document of [manager.load("old"), generateReport(benchmark, loaded)]) {
    const serialized = JSON.parse(JSON.stringify(document))
    for (const key of keys) expect(serialized).not.toHaveProperty(key)
  }
  manager.copyCheckpoint("old", "old-copy", "search")
  const fresh = manager.create("fresh", "fixture", "longmemeval", "judge", "reader", {
    identities: identity(),
  })
  await manager.flush()
  const copy = manager.copyCheckpoint(fresh.runId, "fresh-copy", "search")
  await manager.flush()
  for (const key of keys) {
    expect(manager.load("old-copy")).not.toHaveProperty(key)
    expect(copy[key]).toBe(fresh[key])
  }
})

test("orchestrator records fresh identities, rejects drift before provider init, and preserves old shape", () => {
  const root = temp()
  const result = Bun.spawnSync(
    [
      process.execPath,
      "-e",
      `
    import { mock } from "bun:test";
    import { readFileSync, writeFileSync } from "node:fs";
    const source = ${JSON.stringify(process.cwd())};
    let initialized = 0;
    const benchmark = { name: "locomo", datasetPath: "./synthetic.json", datasetSha256: "a".repeat(64),
      async load() {}, getQuestions: () => [], getQuestionTypes: () => ({}) };
    mock.module(source + "/src/benchmarks/index.ts", () => ({ createBenchmark: () => benchmark }));
    mock.module(source + "/src/providers/index.ts", () => ({ createProvider: () => ({ name: "fixture",
      async initialize() { initialized++; } }) }));
    mock.module(source + "/src/utils/config.ts", () => ({ config: {}, getProviderConfig: () => ({apiKey:""}),
      getJudgeConfig: () => ({apiKey:""}) }));
    const { Orchestrator } = await import(source + "/src/orchestrator/index.ts");
    const options = { provider: "filesystem", benchmark: "locomo", judgeModel: "gpt-4o",
      answeringModel: "gpt-4o", runId: "synthetic", phases: ["report"] };
    const path = "./data/runs/synthetic/checkpoint.json";
    const runner = new Orchestrator();
    await runner.run(options);
    const saved = JSON.parse(readFileSync(path, "utf8"));
    const report = JSON.parse(readFileSync("./data/runs/synthetic/report.json", "utf8"));
    const keys = ${JSON.stringify(keys)};
    for (const key of keys) if (!saved[key] || report[key] !== saved[key]) throw new Error("fresh " + key);
    if (saved.scoringPath !== "openai/default/score-v1") throw new Error("judge name");
    for (const key of ["datasetSha256", "questionSetSha256", "promptVersion", "readerPromptVersion", "scoringPath"]) {
      writeFileSync(path, JSON.stringify({...saved, [key]: "changed"}));
      let refused = false;
      try { await runner.run(options); } catch (error) { refused = error.message.includes(key); }
      if (!refused || initialized !== 1) throw new Error("resume " + key);
    }
    for (const key of keys) delete saved[key];
    writeFileSync(path, JSON.stringify(saved));
    await runner.run(options);
    for (const file of [path, "./data/runs/synthetic/report.json"]) {
      const document = JSON.parse(readFileSync(file, "utf8"));
      for (const key of keys) if (key in document) throw new Error("backfilled " + key);
    }
    console.log("orchestrator identity integration passed");
  `,
    ],
    {
      cwd: root,
      env: {
        ...process.env,
        HERMES_MB_LLM_CLI: "",
        HERMES_MB_DETERMINISTIC_OPERATIONS: "",
        HERMES_MB_ANSWER_PRESENTATION: "",
      },
    }
  )
  if (result.exitCode) throw new Error(result.stderr.toString())
  expect(result.exitCode).toBe(0)
  expect(result.stdout.toString()).toContain("orchestrator identity integration passed")
})

test("rendered provider prompts detect closure changes and separate reader identity", () => {
  const benchmark = {
    ...datasetIdentity("data.json", Buffer.from("[]")),
    getQuestions: () => [],
  } as any
  const provider = (text: string) => ({
    name: "custom",
    prompts: {
      judgePrompt: () => ({ default: text }),
      answerPrompt: () => text,
    },
  })
  const one = runIdentities(benchmark, provider("one"), "judge")
  const two = runIdentities(benchmark, provider("two"), "judge")
  expect(one.promptVersion).not.toBe(two.promptVersion)
  expect(one.readerPromptVersion).not.toBe(two.readerPromptVersion)
  const readerOnly = runIdentities(
    benchmark,
    {
      ...provider("one"),
      prompts: {
        ...provider("one").prompts,
        answerPrompt: "{{question}} {{context}} changed",
      },
    },
    "judge"
  )
  expect(readerOnly.promptVersion).toBe(one.promptVersion)
  expect(readerOnly.readerPromptVersion).not.toBe(one.readerPromptVersion)
})
