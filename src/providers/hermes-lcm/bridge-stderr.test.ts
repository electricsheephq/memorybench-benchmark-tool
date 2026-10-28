import { afterEach, describe, expect, test } from "bun:test"
import { EventEmitter } from "node:events"
import { PassThrough } from "node:stream"
import { logger } from "../../utils/logger"
import { BridgeHandle } from "./index"

const original = { info: logger.info, debug: logger.debug }
afterEach(() => Object.assign(logger, original))

function fakeBridge() {
  const proc = Object.assign(new EventEmitter(), {
    stdin: new PassThrough(),
    stdout: new PassThrough(),
    stderr: new PassThrough(),
  })
  const seen: [string, string][] = []
  logger.info = (message: string) => void seen.push(["info", message])
  logger.debug = (message: string) => void seen.push(["debug", message])
  new BridgeHandle(proc as never, "fx")
  return { proc, seen }
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0))
const MAINTENANCE =
  "[hermes-lcm-bridge] LOCAL maintenance complete=True outcome='summaries=complete:1 chunks=complete:2' wall_s=0.1"

describe("hermes-lcm bridge stderr", () => {
  test("logs the LOCAL maintenance line at info, even when split across chunks", async () => {
    const { proc, seen } = fakeBridge()
    proc.stderr.write(MAINTENANCE.slice(0, 40))
    proc.stderr.write(`${MAINTENANCE.slice(40)}\n[hermes-lcm-bridge] other line\n\n`)
    await flush()
    expect(seen).toEqual([
      ["info", `[hermes-lcm:fx] ${MAINTENANCE}`],
      ["debug", "[hermes-lcm:fx] [hermes-lcm-bridge] other line"],
    ])
  })

  test("logs an unterminated last line when stderr ends", async () => {
    const { proc, seen } = fakeBridge()
    proc.stderr.end(MAINTENANCE.replace("complete=True", "complete=False"))
    await flush()
    expect(seen).toEqual([
      ["info", `[hermes-lcm:fx] ${MAINTENANCE.replace("complete=True", "complete=False")}`],
    ])
  })
})
