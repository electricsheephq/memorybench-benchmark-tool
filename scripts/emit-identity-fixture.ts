import { emitIdentityFixture } from "../src/orchestrator/identity-fixture"

export { emitIdentityFixture }

if (import.meta.main) {
  if (!process.argv[2]) throw new Error("Usage: bun scripts/emit-identity-fixture.ts OUTPUT [A|B]")
  await emitIdentityFixture(process.argv[2], process.argv[3])
}
