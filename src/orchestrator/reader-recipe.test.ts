import { expect, test } from "bun:test"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Frozen checkpoints written by PR #17 (802096d8), with real provider builders.
const legacy = {
  filesystem: {
    datasetPath: "./synthetic.json",
    datasetSha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    questionSetSha256: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    promptVersion: "sha256:25cb037ed030569b00939262257d248daa78300b5b25bdccd204dc1d48d71149",
    readerPromptVersion: "sha256:51546893b48293e1a08ae1923fd14fe0c87d9c7905085b2a1017dd1ecd7fef0b",
    scoringPath: "openai:gpt-4o/default/score-v1",
    answerPresentationMode: "raw_json_v1",
    runId: "probe",
    dataSourceRunId: "probe",
    status: "completed",
    provider: "filesystem",
    benchmark: "locomo",
    judge: "gpt-4o",
    answeringModel: "gpt-4o",
    createdAt: "2026-10-21T19:22:52.794Z",
    updatedAt: "2026-10-21T19:22:52.802Z",
    targetQuestionIds: [],
    questions: {},
  },
  mem0: {
    datasetPath: "./synthetic.json",
    datasetSha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    questionSetSha256: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    promptVersion: "sha256:25cb037ed030569b00939262257d248daa78300b5b25bdccd204dc1d48d71149",
    readerPromptVersion: "sha256:9ef6dfd936c7bc8f0c1ffe09584085e6e85af2424289f1ebc11f2a84b253389e",
    scoringPath: "openai:gpt-4o/default/score-v1",
    answerPresentationMode: "raw_json_v1",
    runId: "probe",
    dataSourceRunId: "probe",
    status: "completed",
    provider: "mem0",
    benchmark: "locomo",
    judge: "gpt-4o",
    answeringModel: "gpt-4o",
    createdAt: "2026-10-21T19:22:53.247Z",
    updatedAt: "2026-10-21T19:22:53.247Z",
    targetQuestionIds: [],
    questions: {},
  },
  zep: {
    datasetPath: "./synthetic.json",
    datasetSha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    questionSetSha256: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    promptVersion: "sha256:f004809bb45d4fd9521a7006325e1fb123342ce069c63d3a9d465f903c0f238d",
    readerPromptVersion: "sha256:2cd38893587ae43d0a836ac00fbb8e8970b7980fad6968343fbb47ac2d9dcfd2",
    scoringPath: "openai:gpt-4o/provider:zep/score-v1",
    answerPresentationMode: "raw_json_v1",
    runId: "probe",
    dataSourceRunId: "probe",
    status: "completed",
    provider: "zep",
    benchmark: "locomo",
    judge: "gpt-4o",
    answeringModel: "gpt-4o",
    createdAt: "2026-10-21T19:22:53.624Z",
    updatedAt: "2026-10-21T19:22:53.624Z",
    targetQuestionIds: [],
    questions: {},
  },
  supermemory: {
    datasetPath: "./synthetic.json",
    datasetSha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    questionSetSha256: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    promptVersion: "sha256:25cb037ed030569b00939262257d248daa78300b5b25bdccd204dc1d48d71149",
    readerPromptVersion: "sha256:5a6d0de9f4fc663d13dc7c537704c5c59a694e60e42224415b0f0fa1cfa5b15b",
    scoringPath: "openai:gpt-4o/default/score-v1",
    answerPresentationMode: "raw_json_v1",
    runId: "probe",
    dataSourceRunId: "probe",
    status: "completed",
    provider: "supermemory",
    benchmark: "locomo",
    judge: "gpt-4o",
    answeringModel: "gpt-4o",
    createdAt: "2026-10-21T19:22:54.208Z",
    updatedAt: "2026-10-21T19:22:54.209Z",
    targetQuestionIds: [],
    questions: {},
  },
  cortex: {
    datasetPath: "./synthetic.json",
    datasetSha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    questionSetSha256: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    promptVersion: "sha256:25cb037ed030569b00939262257d248daa78300b5b25bdccd204dc1d48d71149",
    readerPromptVersion: "sha256:5d425acc4cf683ad4598b65ab496256013fa842c93fa6ed9b378f1c7d479e692",
    scoringPath: "openai:gpt-4o/default/score-v1",
    answerPresentationMode: "raw_json_v1",
    runId: "probe",
    dataSourceRunId: "probe",
    status: "completed",
    provider: "cortex",
    benchmark: "locomo",
    judge: "gpt-4o",
    answeringModel: "gpt-4o",
    createdAt: "2026-10-21T19:22:54.631Z",
    updatedAt: "2026-10-21T19:22:54.632Z",
    targetQuestionIds: [],
    questions: {},
  },
  "hermes-lcm": {
    datasetPath: "./synthetic.json",
    datasetSha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    questionSetSha256: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    promptVersion: "sha256:25cb037ed030569b00939262257d248daa78300b5b25bdccd204dc1d48d71149",
    readerPromptVersion: "sha256:6c9794943db1d27043c61e1a9df96743f90ac858e1cdcb4401804035769a7f58",
    scoringPath: "openai:gpt-4o/default/score-v1",
    answerPresentationMode: "raw_json_v1",
    runId: "probe",
    dataSourceRunId: "probe",
    status: "completed",
    provider: "hermes-lcm",
    benchmark: "locomo",
    judge: "gpt-4o",
    answeringModel: "gpt-4o",
    createdAt: "2026-10-21T19:22:55.223Z",
    updatedAt: "2026-10-21T19:22:55.224Z",
    targetQuestionIds: [],
    questions: {},
  },
  rag: {
    datasetPath: "./synthetic.json",
    datasetSha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    questionSetSha256: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    promptVersion: "sha256:25cb037ed030569b00939262257d248daa78300b5b25bdccd204dc1d48d71149",
    readerPromptVersion: "sha256:0bfbff641a4a0bf47972ad8f1be5c2456712538a4aa3aa74300fd21da509ec51",
    scoringPath: "openai:gpt-4o/default/score-v1",
    answerPresentationMode: "raw_json_v1",
    runId: "probe",
    dataSourceRunId: "probe",
    status: "completed",
    provider: "rag",
    benchmark: "locomo",
    judge: "gpt-4o",
    answeringModel: "gpt-4o",
    createdAt: "2026-10-21T19:22:55.999Z",
    updatedAt: "2026-10-21T19:22:55.999Z",
    targetQuestionIds: [],
    questions: {},
  },
  cli: {
    datasetPath: "./synthetic.json",
    datasetSha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    questionSetSha256: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    promptVersion: "sha256:25cb037ed030569b00939262257d248daa78300b5b25bdccd204dc1d48d71149",
    readerPromptVersion: "sha256:51546893b48293e1a08ae1923fd14fe0c87d9c7905085b2a1017dd1ecd7fef0b",
    scoringPath: "cli:codex:codex default (un-pinned)/default/score-v1",
    answerPresentationMode: "raw_json_v1",
    runId: "probe",
    dataSourceRunId: "probe",
    status: "completed",
    provider: "filesystem",
    benchmark: "locomo",
    judge: "gpt-4o",
    answeringModel: "gpt-4o",
    createdAt: "2026-10-21T19:22:56.650Z",
    updatedAt: "2026-10-21T19:22:56.956Z",
    targetQuestionIds: [],
    questions: {},
  },
}
const source = process.cwd()
function probe(body: string) {
  const root = mkdtempSync(join(tmpdir(), "reader-recipe-"))
  try {
    const result = Bun.spawnSync([process.execPath, "-e", body], {
      cwd: root,
      env: { ...process.env, HERMES_MB_LLM_CLI: "" },
      timeout: 10000,
    })
    if (result.exitCode) throw new Error(result.stdout.toString() + result.stderr.toString())
    expect(result.exitCode).toBe(0)
  } finally {
    rmSync(root, { recursive: true })
  }
}
function setup(name: string) {
  return `
  import { mock } from "bun:test";
  import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
  const source = ${JSON.stringify(source)};
  globalThis.fetch = () => { throw new Error("network forbidden"); };
  let initialized = 0;
  const name = ${JSON.stringify(name)};
  const original = Object.values(await import(source + "/src/providers/" + name + "/prompts.ts"))
    .find(value => value?.answerPrompt);
  let prompts = {...original};
  const benchmark = {name:"locomo",datasetPath:"./synthetic.json",datasetSha256:"a".repeat(64),
    async load(){},getQuestions:()=>[],getHaystackSessions:()=>[],getQuestionTypes:()=>({})};
  mock.module(source + "/src/benchmarks/index.ts",()=>({createBenchmark:()=>benchmark,getAvailableBenchmarks:()=>["locomo"]}));
  mock.module(source + "/src/providers/index.ts",()=>({createProvider:()=>({name,prompts,
    async initialize(){initialized++}}),getAvailableProviders:()=>[name]}));
  mock.module(source + "/src/judges/index.ts",()=>({createJudge:()=>({async initialize(){}})}));
  mock.module(source + "/src/utils/config.ts",()=>({config:{},getProviderConfig:()=>({apiKey:""}),getJudgeConfig:()=>({apiKey:""})}));
  const { Orchestrator } = await import(source + "/src/orchestrator/index.ts");
  const { runIdentities } = await import(source + "/src/orchestrator/identities.ts");
  const warnings = [];
  const { logger } = await import(source + "/src/utils/logger.ts");
  logger.warn = message => warnings.push(message);
  const runner = new Orchestrator();
  const options = {provider:name,benchmark:"locomo",judgeModel:"gpt-4o",answeringModel:"gpt-4o",runId:"probe",phases:[]};
  const path = "./data/runs/probe/checkpoint.json";
  const read = () => JSON.parse(readFileSync(path,"utf8"));
  async function refuse() {
    const before = readFileSync(path,"utf8"), count = initialized;
    let error = "";
    try { await runner.run({...options,phases:["answer"]}); } catch(e) { error=e.message; }
    if (!error.includes("identity mismatch: readerPromptVersion") || initialized !== count || readFileSync(path,"utf8") !== before)
      throw new Error("prompt drift accepted or checkpoint/provider touched: " + error);
  }
`
}
for (const [key, checkpoint] of Object.entries({ ...legacy, command: legacy.filesystem })) {
  test(`PR17 ${key} checkpoint accepts unchanged answer resume and preserves stored identity`, () => {
    probe(
      setup(checkpoint.provider) +
        `
      const saved = ${JSON.stringify(checkpoint)};
      mkdirSync("./data/runs/probe",{recursive:true});writeFileSync(path,JSON.stringify(saved));
      ${key === "cli" ? 'process.env.HERMES_MB_LLM_CLI="codex";' : ""}
      ${key === "command" ? 'const {runCommand}=await import(source+"/src/cli/commands/run.ts");await runCommand(["-r","probe","-f","answer"]);' : 'await runner.run({...options,phases:["answer"]});'}
      if (read().readerPromptVersion !== saved.readerPromptVersion || initialized !== 1) throw new Error("legacy rewrite/init");
      if (warnings.filter(message => message === "legacy checkpoint: prompt identity covers the main prompt renders only; start a fresh run for full coverage").length !== 1)
        throw new Error("legacy warning must appear exactly once");
    `
    )
  })
}
test("PR17 checkpoint still refuses a changed main prompt before provider init", () => {
  probe(
    setup("filesystem") +
      `
    mkdirSync("./data/runs/probe",{recursive:true});writeFileSync(path,JSON.stringify(${JSON.stringify(legacy.filesystem)}));
    const build = prompts.answerPrompt;prompts.answerPrompt=(...args)=>build(...args)+"changed main prompt";
    await refuse();
  `
  )
})
for (const [name, before, after] of [
  ["cortex", 'subject || "?"', 'subject || "unknown subject"'],
  ["cortex", 'predicate || "?"', 'predicate || "unknown predicate"'],
  ["cortex", 'object || "?"', 'object || "unknown object"'],
  ["zep", 'result.name || "Unknown"', 'result.name || "Unnamed entity"'],
  ["zep", 'result.summary || ""', 'result.summary || "No summary"'],
  ["supermemory", "chunk.position ?? 0", "chunk.position ?? 100"],
]) {
  test(`fallback edit changes version and refuses answer: ${before}`, () => {
    probe(
      setup(name!) +
        `
      await runner.run(options);
      const old = read().readerPromptVersion;
      await runner.run({...options,phases:["answer"]});
      const text=readFileSync(source+"/src/providers/"+name+"/prompts.ts","utf8");
      const changed=text.replace(${JSON.stringify(before)},${JSON.stringify(after)});
      if(text===changed) throw new Error("mutation did not apply");
      writeFileSync("./mutant.ts",changed);
      const mutant=await import(process.cwd()+"/mutant.ts");
      prompts={...prompts,answerPrompt:Object.values(mutant).find(value=>value?.answerPrompt).answerPrompt};
      if(runIdentities(benchmark,{name,prompts},"openai:gpt-4o").readerPromptVersion===old) throw new Error("version unchanged");
      await refuse();
    `
    )
  })
}

test("shared session is serialized once while both questions retain its id", () => {
  probe(
    setup("filesystem") +
      `
    let serializations=0;
    const session={sessionId:"shared",get messages(){serializations++;return [{role:"user",content:"Synthetic"}]}};
    benchmark.getQuestions=()=>["q1","q2"].map(questionId=>({questionId,question:"Q",groundTruth:"A",questionType:"single-hop",haystackSessionIds:["shared"]}));
    benchmark.getHaystackSessions=()=>[session];
    runIdentities(benchmark,{name,prompts},"openai:gpt-4o");
    if(serializations!==1) throw new Error("shared session serialized " + serializations + " times");
  `
  )
})


test("tagged fresh runs and resumes never print the legacy warning", () => {
  probe(
    setup("filesystem") +
      `
    await runner.run(options);
    await runner.run({...options,phases:["answer"]});
    if (warnings.some(message => message.includes("legacy checkpoint:"))) throw new Error("tagged warning");
  `
  )
})

// Adapted from Astra's session-collision-head probe; checkpoints are synthetic.
test("conflicting shared session content from either question refuses before init without writes", () => {
  probe(
    setup("filesystem") +
      `
    benchmark.getQuestions=()=>["q1","q2"].map(questionId=>({questionId,question:"Q",groundTruth:"A",questionType:"single-hop",haystackSessionIds:["shared"]}));
    const perQuestion={q1:[{sessionId:"shared",messages:[{role:"user",content:"A"}]}],q2:[{sessionId:"shared",messages:[{role:"user",content:"B"}]}]};
    benchmark.getHaystackSessions=id=>perQuestion[id];
    await runner.run(options);
    await runner.run(options);
    const before=readFileSync(path,"utf8"), count=initialized;
    for (const [id, original] of [["q1","A"],["q2","B"]]) {
      perQuestion[id][0].messages[0].content="CHANGED " + original;
      let error="";
      try { await runner.run(options); } catch(e) { error=e.message; }
      if (!error.includes("identity mismatch: normalizedInputsSha256") || initialized!==count || readFileSync(path,"utf8")!==before)
        throw new Error(id + " content change accepted or checkpoint/provider touched: " + error);
      perQuestion[id][0].messages[0].content=original;
    }
  `
  )
})

test("equal shared content in distinct objects is deduplicated", () => {
  probe(
    setup("filesystem") +
      `
    benchmark.getQuestions=()=>["q1","q2"].map(questionId=>({questionId,question:"Q",groundTruth:"A",questionType:"single-hop",haystackSessionIds:["shared"]}));
    const session={sessionId:"shared",messages:[{role:"user",content:"Synthetic"}]};
    benchmark.getHaystackSessions=()=>[session];
    const shared=runIdentities(benchmark,{name,prompts},"openai:gpt-4o").normalizedInputsSha256;
    benchmark.getHaystackSessions=()=>[JSON.parse(JSON.stringify(session))];
    const copied=runIdentities(benchmark,{name,prompts},"openai:gpt-4o").normalizedInputsSha256;
    if (shared!==copied) throw new Error("equal-content objects serialized more than once");
  `
  )
})

// R3: the global A/B variant set stays unchanged when q3 switches A -> B.
test("A B A to A B B refuses before provider init with byte-identical checkpoint", () => {
  probe(
    setup("filesystem") +
      `
    benchmark.getQuestions=()=>["q1","q2","q3"].map(questionId=>({questionId,question:"Q",groundTruth:"A",questionType:"single-hop",haystackSessionIds:["s"]}));
    const a={sessionId:"s",messages:[{role:"user",content:"A"}]};
    const b={sessionId:"s",messages:[{role:"user",content:"B"}]};
    const perQuestion={q1:[a],q2:[b],q3:[a]};
    benchmark.getHaystackSessions=id=>perQuestion[id];
    await runner.run(options);
    await runner.run({...options,phases:["answer"]});
    const before=readFileSync(path), count=initialized;
    perQuestion.q3=[b];
    let error="";
    try { await runner.run({...options,phases:["answer"]}); } catch(e) { error=e.message; }
    if (!error.includes("identity mismatch: normalizedInputsSha256") || initialized!==count || !readFileSync(path).equals(before))
      throw new Error("A B A -> A B B accepted or checkpoint/provider touched: " + error);
  `
  )
})

for (const [name, label, before, after] of [
  ["cortex", "date precedence", "claim.metadata?.event_date ||\n    claim.metadata?.date ||\n    claim.metadata?.timestamp ||\n    claim.metadata?.session_date", "claim.metadata?.session_date ||\n    claim.metadata?.timestamp ||\n    claim.metadata?.date ||\n    claim.metadata?.event_date"],
  ["mem0", "date/timestamp precedence", "metadata.date || metadata.timestamp", "metadata.timestamp || metadata.date"],
  ["cortex", "memory/fact/content/text precedence", "stringifyValue(claim.memory) ||\n    stringifyValue(claim.fact) ||\n    stringifyValue(claim.content) ||\n    stringifyValue(claim.text)", "stringifyValue(claim.text) ||\n    stringifyValue(claim.content) ||\n    stringifyValue(claim.fact) ||\n    stringifyValue(claim.memory)"],
  ["zep", "name/summary selection", 'const name = result.name || "Unknown"\n      const summary = result.summary || ""', 'const name = result.summary || "Unknown"\n      const summary = result.name || ""'],
  ["zep", "fact/JSON precedence", "result.fact || JSON.stringify(r)", "JSON.stringify(r) || result.fact"],
  ["mem0", "memory/JSON precedence", "mem.memory || JSON.stringify(r)", "JSON.stringify(r) || mem.memory"],
]) {
  test(`precedence reversal changes tagged reader and refuses answer: ${name} ${label}`, () => {
    probe(
      setup(name!) +
        `
      await runner.run(options);
      const old=read().readerPromptVersion;
      if (!old.startsWith("rpv2:")) throw new Error("expected tagged reader");
      await runner.run({...options,phases:["answer"]});
      const text=readFileSync(source+"/src/providers/"+name+"/prompts.ts","utf8");
      const changed=text.replace(${JSON.stringify(before)},${JSON.stringify(after)});
      if(text===changed) throw new Error("mutation did not apply");
      writeFileSync("./precedence-mutant.ts",changed);
      const mutant=await import(process.cwd()+"/precedence-mutant.ts");
      prompts={...prompts,answerPrompt:Object.values(mutant).find(value=>value?.answerPrompt).answerPrompt};
      if(runIdentities(benchmark,{name,prompts},"openai:gpt-4o").readerPromptVersion===old) throw new Error("version unchanged");
      await refuse();
    `
    )
  })
}
