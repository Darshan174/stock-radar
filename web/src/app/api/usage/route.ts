import { NextResponse } from "next/server"
import { promises as fs } from "fs"
import path from "path"
import os from "os"

// API limits configuration (should match usage_tracker.py)
const API_LIMITS: Record<string, { limit: number | null; period: string; unit: string }> = {
  groq: { limit: 14400, period: "daily", unit: "requests" },
  gemini: { limit: 1500, period: "daily", unit: "requests" },
  cohere: { limit: 1000, period: "monthly", unit: "embeds" },
  finnhub: { limit: 30000, period: "daily", unit: "calls" },
  ollama: { limit: null, period: "unlimited", unit: "calls" },
}

export async function GET() {
  try {
    const usagePath = path.join(os.homedir(), ".stock-radar", "usage.json")

    try {
      const data = await fs.readFile(usagePath, "utf-8")
      const rawUsage = JSON.parse(data)

      // Transform the data to include both counts and tokens with limits
      const services: Record<string, {
        count: number
        tokens: number
        limit: number | null
        period: string
        unit: string
        percentage: number
        last_reset: string | null
      }> = {}

      for (const [serviceName, config] of Object.entries(API_LIMITS)) {
        const serviceData = rawUsage.services?.[serviceName] || { count: 0, tokens: 0 }
        const count = serviceData.count || 0
        const tokens = serviceData.tokens || 0
        const limit = config.limit
        const percentage = limit ? Math.round((count / limit) * 100 * 100) / 100 : 0

        services[serviceName] = {
          count,
          tokens,
          limit,
          period: config.period,
          unit: config.unit,
          percentage,
          last_reset: serviceData.last_reset || null,
        }
      }

      return NextResponse.json({
        services,
        created_at: rawUsage.created_at || null,
      })
    } catch {
      // Return default values if file doesn't exist
      const services: Record<string, {
        count: number
        tokens: number
        limit: number | null
        period: string
        unit: string
        percentage: number
        last_reset: string | null
      }> = {}

      for (const [serviceName, config] of Object.entries(API_LIMITS)) {
        services[serviceName] = {
          count: 0,
          tokens: 0,
          limit: config.limit,
          period: config.period,
          unit: config.unit,
          percentage: 0,
          last_reset: null,
        }
      }

      return NextResponse.json({ services, created_at: null })
    }
  } catch (error) {
    console.error("Error reading usage data:", error)
    return NextResponse.json({ error: "Failed to read usage data" }, { status: 500 })
  }
}
