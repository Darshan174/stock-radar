import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol } from "@/lib/input-validation"

const execAsync = promisify(exec)

export async function handleFundamentals(request: NextRequest): Promise<NextResponse> {
  const { searchParams } = new URL(request.url)

  const symbolCheck = validateSymbol(searchParams.get("symbol"))
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const symbol = symbolCheck.value

  const projectRoot = path.resolve(process.cwd(), "..")
  const scriptPath = path.join(projectRoot, "scripts", "get_fundamentals.py")

  const { stdout, stderr } = await execAsync(
    `python3 "${scriptPath}" "${symbol}"`,
    {
      timeout: 30000,
      env: { ...process.env, SR_SYMBOL: symbol },
    }
  )

  if (stderr) {
    console.error("Python stderr:", stderr)
  }

  const data = JSON.parse(stdout.trim())
  return NextResponse.json(data)
}

export async function GET(request: NextRequest) {
  return withX402(request, "/api/fundamentals", handleFundamentals)
}
