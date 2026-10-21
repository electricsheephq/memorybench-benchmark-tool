import { expect, test } from "bun:test"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

for (const judge of ["sonnet-4.5", "gemini-2.5-pro"]) {
  test(`search/ingest CLI resumes recorded ${judge}, evaluate rejects changed judge`, () => {
    const root = mkdtempSync(join(tmpdir(), "resume-judge-"))
    try {
      const result = Bun.spawnSync(
        [
          process.execPath,
          "-e",
          `
        import { mock } from "bun:test";
        import { readFileSync } from "node:fs";
        const source = ${JSON.stringify(process.cwd())};
        let initialized = 0;
        const benchmark = {name:"locomo", datasetPath:"./synthetic.json", datasetSha256:"a".repeat(64),
          async load(){}, getQuestions:()=>[], getQuestionTypes:()=>({})};
        mock.module(source + "/src/benchmarks/index.ts", () => ({createBenchmark:()=>benchmark,
          getAvailableBenchmarks:()=>["locomo"]}));
        mock.module(source + "/src/providers/index.ts", () => ({createProvider:()=>({name:"fixture",
          async initialize(){initialized++}}), getAvailableProviders:()=>["filesystem"]}));
        mock.module(source + "/src/utils/config.ts", () => ({config:{},
          getProviderConfig:()=>({apiKey:""}), getJudgeConfig:()=>({apiKey:""})}));
        const { Orchestrator } = await import(source + "/src/orchestrator/index.ts");
        const runner = new Orchestrator();
        const options = {provider:"filesystem",benchmark:"locomo",judgeModel:${JSON.stringify(judge)},
          answeringModel:"gpt-4o",runId:"r1",phases:[]};
        await runner.run(options);
        const path = "./data/runs/r1/checkpoint.json";
        const saved = JSON.parse(readFileSync(path,"utf8"));
        const { searchCommand } = await import(source + "/src/cli/commands/search.ts");
        const { ingestCommand } = await import(source + "/src/cli/commands/ingest.ts");
        await searchCommand(["-r","r1"]);
        await ingestCommand(["-r","r1"]);
        if(initialized !== 3) throw new Error("resume did not initialize provider");
        const resumed = JSON.parse(readFileSync(path,"utf8"));
        if(resumed.judge !== saved.judge || resumed.scoringPath !== saved.scoringPath) throw new Error("judge drift");
        const before = readFileSync(path,"utf8");
        let refused = false;
        try { await runner.run({...options,judgeModel:"gpt-4o",phases:["evaluate"]}) }
        catch(e) { refused = e.message.includes("scoringPath") }
        if(!refused || initialized !== 3 || before !== readFileSync(path,"utf8")) throw new Error("evaluate mismatch not refused before work");
        console.log("resume and evaluate guard passed");
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
          timeout: 10000,
        }
      )
      if (result.exitCode) throw new Error(result.stderr.toString())
      expect(result.exitCode).toBe(0)
      expect(result.stdout.toString()).toContain("resume and evaluate guard passed")
    } finally {
      rmSync(root, { recursive: true })
    }
  })
}
