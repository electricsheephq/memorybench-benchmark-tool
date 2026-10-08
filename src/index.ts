import { cli } from "./cli"

const args = process.argv.slice(2)
// A failed run exits non-zero, so a watchdog never records success for a run without a report (lcm-x #976).
cli(args).catch((error) => {
  console.error(error)
  process.exitCode = 1
})
