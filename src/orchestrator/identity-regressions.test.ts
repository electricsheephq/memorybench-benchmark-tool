import { expect, test } from "bun:test"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Adapted from the independent review's constructor, copy and resume probes.
// Each subprocess gets isolated module mocks and synthetic checkpoints only.
function probe(body: string): string {
  const root = mkdtempSync(join(tmpdir(), "identity-regression-"))
  try {
    const result = Bun.spawnSync([process.execPath, "-e", body], {
      cwd: root,
      env: {
        ...process.env,
        HERMES_MB_LLM_CLI: "",
        HERMES_MB_DETERMINISTIC_OPERATIONS: "",
        HERMES_MB_ANSWER_PRESENTATION: "",
      },
      timeout: 10000,
    })
    if (result.exitCode) throw new Error(result.stderr.toString())
    expect(result.exitCode).toBe(0)
    return result.stdout.toString()
  } finally {
    rmSync(root, { recursive: true })
  }
}

const source = JSON.stringify(process.cwd())
const setup = `
  import { mock } from "bun:test";
  import { readFileSync, writeFileSync } from "node:fs";
  const source = ${source};
  globalThis.fetch = () => { throw new Error("network forbidden"); };
  let initialized = 0;
  const benchmark = {name:"locomo", datasetPath:"./synthetic.json", datasetSha256:"a".repeat(64),
    async load(){}, getQuestions:()=>[], getQuestionTypes:()=>({})};
  mock.module(source + "/src/benchmarks/index.ts", () => ({createBenchmark:()=>benchmark,
    getAvailableBenchmarks:()=>["locomo"]}));
  mock.module(source + "/src/providers/index.ts", () => ({createProvider:()=>({name:"filesystem",
    async initialize(){initialized++}}), getAvailableProviders:()=>["filesystem"]}));
  mock.module(source + "/src/judges/index.ts", () => ({createJudge:()=>({async initialize(){}})}));
  mock.module(source + "/src/utils/config.ts", () => ({config:{},
    getProviderConfig:()=>({apiKey:""}), getJudgeConfig:()=>({apiKey:""})}));
  const { Orchestrator, CheckpointManager } = await import(source + "/src/orchestrator/index.ts");
  const { getPhasesFromPhase } = await import(source + "/src/types/checkpoint.ts");
  const runner = new Orchestrator();
  const manager = new CheckpointManager();
  const options = {provider:"filesystem",benchmark:"locomo",judgeModel:"sonnet-4.5",
    answeringModel:"gpt-4o",runId:"source",phases:["report"]};
  const path = id => "./data/runs/" + id + "/checkpoint.json";
  const read = id => JSON.parse(readFileSync(path(id),"utf8"));
  async function refusesUnchanged(runId, phases, field, judgeModel = options.judgeModel) {
    const before = readFileSync(path(runId),"utf8");
    const count = initialized;
    let refused = false;
    try { await runner.run({...options,runId,phases,judgeModel}); }
    catch(e) { refused = e.message.includes("identity mismatch: " + field); }
    if(!refused || initialized !== count || before !== readFileSync(path(runId),"utf8"))
      throw new Error(field + " mismatch not refused before provider init with unchanged checkpoint");
  }
`

test("every registered provider renders both identities stably across two processes", () => {
  const script = `
    globalThis.fetch = () => { throw new Error("network forbidden"); };
    const { createProvider, getAvailableProviders } = await import(${source} + "/src/providers/index.ts");
    const { runIdentities } = await import(${source} + "/src/orchestrator/identities.ts");
    const benchmark = {datasetPath:"./synthetic.json",datasetSha256:"a".repeat(64),getQuestions:()=>[]};
    const identities = {};
    for(const name of getAvailableProviders()) {
      const value = runIdentities(benchmark,createProvider(name),"openai");
      for(const field of ["promptVersion","readerPromptVersion"])
        if(!/^sha256:[a-f0-9]{64}$/.test(value[field])) throw new Error(name + ": " + field);
      identities[name] = value;
    }
    console.log(JSON.stringify(identities));
  `
  const first = JSON.parse(probe(script))
  const second = JSON.parse(probe(script))
  expect(first).toEqual(second)
  expect(first).toHaveProperty("rag")
})

for (const fromPhase of ["answer", "search", "indexing"]) {
  test(`advanced ${fromPhase} copy accepts a cross-provider judge override and stamps its identity`, () => {
    expect(
      probe(
        setup +
          `
        await runner.run(options);
        if(read("source").scoringPath !== "anthropic/default/score-v1") throw new Error("source identity");
        manager.copyCheckpoint("source","copy",${JSON.stringify(fromPhase)},{judge:"gpt-4o"});
        await manager.flush("copy");
        await runner.run({...options,runId:"copy",judgeModel:"gpt-4o",
          phases:getPhasesFromPhase(${JSON.stringify(fromPhase)})});
        const copied = read("copy");
        if(copied.judge !== "gpt-4o" || copied.scoringPath !== "openai/default/score-v1" ||
          copied.promptVersion !== read("source").promptVersion) throw new Error("copy identity not stamped");
        await refusesUnchanged("copy",["evaluate"],"scoringPath");
        manager.copyCheckpoint("source","no-override",${JSON.stringify(fromPhase)});
        await manager.flush("no-override");
        if(read("no-override").scoringPath !== read("source").scoringPath ||
          read("no-override").promptVersion !== read("source").promptVersion) throw new Error("source identity lost");
        await refusesUnchanged("no-override",["evaluate"],"scoringPath","gpt-4o");
        console.log("copy override and preserved guard passed");
      `
      )
    ).toContain("copy override and preserved guard passed")
  })
}

for (const command of ["search", "ingest"]) {
  test(`${command} -r accepts a changed CLI backend while evaluate and data guards still refuse`, () => {
    expect(
      probe(
        setup +
          `
        process.env.HERMES_MB_LLM_CLI = "codex";
        await runner.run(options);
        const saved = read("source");
        if(saved.scoringPath !== "cli/default/score-v1") throw new Error("CLI creation identity");
        delete process.env.HERMES_MB_LLM_CLI;
        const { ${command}Command } = await import(source + "/src/cli/commands/${command}.ts");
        await ${command}Command(["-r","source"]);
        if(initialized !== 2 || read("source").scoringPath !== saved.scoringPath)
          throw new Error("non-scoring resume failed or changed judge identity");
        await refusesUnchanged("source",["evaluate"],"scoringPath");
        for(const field of ["datasetSha256","questionSetSha256"]) {
          writeFileSync(path("source"),JSON.stringify({...saved,[field]:"tampered"}));
          await refusesUnchanged("source",${JSON.stringify(command === "search" ? ["search"] : ["ingest", "indexing"])},field);
        }
        writeFileSync(path("source"),JSON.stringify({...saved,promptVersion:"different judge prompt"}));
        await ${command}Command(["-r","source"]);
        process.env.HERMES_MB_LLM_CLI = "codex";
        await refusesUnchanged("source",["evaluate"],"promptVersion");
        console.log("CLI resume and scoring/data guards passed");
      `
      )
    ).toContain("CLI resume and scoring/data guards passed")
  })
}
