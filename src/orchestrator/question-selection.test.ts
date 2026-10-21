import { expect, test } from "bun:test"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

test("checkpoint/report hash ordered selections and refuse changed ids before provider work", () => {
  const root = mkdtempSync(join(tmpdir(), "selection-"))
  try {
    const result = Bun.spawnSync(
      [
        process.execPath,
        "-e",
        `
      import { mock } from "bun:test";
      import { createHash } from "node:crypto";
      import { readFileSync, writeFileSync } from "node:fs";
      const source = ${JSON.stringify(process.cwd())};
      let initialized = 0;
      const questions = ["c","a","b","d"].map((questionId,i) => ({questionId,
        questionType:i%2 ? "multi-session" : "single-session-user", question:"Synthetic",
        groundTruth:"Synthetic", haystackSessionIds:[]}));
      mock.module(source + "/src/benchmarks/index.ts", () => ({createBenchmark:()=>({
        name:"longmemeval",datasetPath:"./synthetic.json",datasetSha256:"a".repeat(64),
        async load(){},getQuestions:()=>questions,getHaystackSessions:()=>[],getQuestionTypes:()=>({})})}));
      mock.module(source + "/src/providers/index.ts", () => ({createProvider:()=>({name:"fixture",
        async initialize(){initialized++}})}));
      mock.module(source + "/src/utils/config.ts", () => ({config:{},
        getProviderConfig:()=>({apiKey:""}),getJudgeConfig:()=>({apiKey:""})}));
      const { Orchestrator } = await import(source + "/src/orchestrator/index.ts");
      for (const [selection,expected] of [
        [{limit:2},["c","a"]],
        [{sampling:{mode:"sample",sampleType:"consecutive",perCategory:1}},["c","a"]],
        [{questionIds:["d","b"]},["d","b"]],
        [{},["c","a","b","d"]],
      ]) {
        const runner = new Orchestrator();
        const options = {provider:"filesystem",benchmark:"longmemeval",judgeModel:"gpt-4o",
          runId:"r"+initialized,phases:["report"],...selection};
        await runner.run(options);
        const path = "data/runs/" + options.runId + "/checkpoint.json";
        const cp = JSON.parse(readFileSync(path,"utf8"));
        const report = JSON.parse(readFileSync("data/runs/"+options.runId+"/report.json","utf8"));
        const expectedHash = createHash("sha256").update(JSON.stringify(expected)).digest("hex");
        if(JSON.stringify(cp.targetQuestionIds)!==JSON.stringify(expected) || cp.questionSetSha256!==expectedHash || report.questionSetSha256!==expectedHash) throw new Error("selection identity");
        await runner.run(options);
        const count = initialized;
        writeFileSync(path,JSON.stringify({...cp,targetQuestionIds:[...cp.targetQuestionIds].reverse()}));
        const before = readFileSync(path,"utf8");
        let refused = false;
        try {await runner.run(options)} catch(e) {refused=e.message.includes("questionSetSha256")}
        if(!refused || initialized!==count || before!==readFileSync(path,"utf8")) throw new Error("selection mismatch accepted");
      }
      console.log("ordered selection guard passed");
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
    expect(result.stdout.toString()).toContain("ordered selection guard passed")
  } finally {
    rmSync(root, { recursive: true })
  }
})
