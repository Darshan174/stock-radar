import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"

const execAsync = promisify(exec)

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url)
    const symbol = searchParams.get("symbol")

    if (!symbol) {
        return NextResponse.json({ error: "Symbol is required" }, { status: 400 })
    }

    try {
        // Get the project root directory
        const projectRoot = path.resolve(process.cwd(), "..")
        const scriptPath = path.join(projectRoot, "scripts", "get_fundamentals.py")

        // Execute Python script
        const { stdout, stderr } = await execAsync(
            `python3 "${scriptPath}" "${symbol}"`,
            { timeout: 30000 }
        )

        if (stderr) {
            console.error("Python stderr:", stderr)
        }

        const data = JSON.parse(stdout.trim())
        return NextResponse.json(data)

    } catch (error: any) {
        console.error("Error fetching fundamentals:", error)

        // Return partial data with error
        return NextResponse.json({
            symbol,
            error: error.message || "Failed to fetch fundamentals"
        })
    }
}
