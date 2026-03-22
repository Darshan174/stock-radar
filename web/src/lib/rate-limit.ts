import { NextRequest, NextResponse } from "next/server"

type BucketConfig = {
  key: string
  limit: number
  windowSeconds: number
}

type CounterRecord = {
  count: number
  resetAt: number
}

const inMemoryCounters = new Map<string, CounterRecord>()

function getClientIp(request: NextRequest): string {
  const forwarded = request.headers.get("x-forwarded-for")
  if (forwarded) {
    const first = forwarded.split(",")[0]?.trim()
    if (first && first !== "unknown") return first
  }

  const cfIp = request.headers.get("cf-connecting-ip")
  if (cfIp) return cfIp

  const realIp = request.headers.get("x-real-ip")
  if (realIp) return realIp

  // Local/dev: derive a key from the user-agent so different browsers
  // don't collapse into one shared bucket.
  const ua = request.headers.get("user-agent") || ""
  let hash = 0
  for (let i = 0; i < ua.length; i++) {
    hash = ((hash << 5) - hash + ua.charCodeAt(i)) | 0
  }
  return `local-${hash.toString(36)}`
}

function shouldBypass(request: NextRequest): boolean {
  const internalHeader = request.headers.get("X-Internal-Key")
  const expected = process.env.INTERNAL_API_KEY
  if (internalHeader && expected && internalHeader === expected) {
    return true
  }

  return false
}

async function incrementUpstash(key: string, windowSeconds: number): Promise<number | null> {
  const restUrl = process.env.UPSTASH_REDIS_REST_URL
  const restToken = process.env.UPSTASH_REDIS_REST_TOKEN

  if (!restUrl || !restToken) {
    return null
  }

  const response = await fetch(`${restUrl}/pipeline`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${restToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify([
      ["INCR", key],
      ["EXPIRE", key, String(windowSeconds)],
    ]),
    cache: "no-store",
  })

  if (!response.ok) {
    return null
  }

  const body = (await response.json()) as Array<{ result?: number | string }> | undefined
  if (!Array.isArray(body) || body.length === 0) {
    return null
  }

  const value = body[0]?.result
  if (typeof value === "number") return value
  if (typeof value === "string") return Number.parseInt(value, 10)
  return null
}

function incrementInMemory(key: string, windowSeconds: number): number {
  const now = Date.now()
  const resetAt = now + windowSeconds * 1000
  const existing = inMemoryCounters.get(key)

  if (!existing || existing.resetAt < now) {
    inMemoryCounters.set(key, { count: 1, resetAt })
    return 1
  }

  existing.count += 1
  return existing.count
}

export async function enforceRateLimit(
  request: NextRequest,
  bucket: BucketConfig,
): Promise<NextResponse | null> {
  if (shouldBypass(request)) {
    console.info(`[rate-limit] bypassed for internal request on ${request.nextUrl.pathname}`)
    return null
  }

  const ip = getClientIp(request)
  const windowId = Math.floor(Date.now() / (bucket.windowSeconds * 1000))
  const redisKey = `ratelimit:${bucket.key}:${ip}:${windowId}`

  let count: number
  try {
    const remoteCount = await incrementUpstash(redisKey, bucket.windowSeconds)
    count = remoteCount ?? incrementInMemory(redisKey, bucket.windowSeconds)
  } catch {
    count = incrementInMemory(redisKey, bucket.windowSeconds)
  }

  if (count <= bucket.limit) {
    return null
  }

  return NextResponse.json(
    {
      error: "Rate limit exceeded",
      code: 429,
      retryAfterSeconds: bucket.windowSeconds,
    },
    {
      status: 429,
      headers: {
        "Retry-After": String(bucket.windowSeconds),
      },
    },
  )
}

export const RATE_BUCKETS = {
  free: {
    key: "free",
    limit: 30,
    windowSeconds: 60,
  },
  paid: {
    key: "paid",
    limit: 10,
    windowSeconds: 60,
  },
  analyzeJobs: {
    key: "analyze-jobs",
    limit: 10,
    windowSeconds: 60,
  },
  jobStatus: {
    key: "job-status",
    limit: 120,
    windowSeconds: 60,
  },
} as const
