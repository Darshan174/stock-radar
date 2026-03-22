import fs from "node:fs"
import path from "node:path"

import { NextResponse } from "next/server"

const LOCAL_DEV_BACKEND_API_KEY = "stock-radar-local-dev-key"

export class BackendProxyError extends Error {
  status: number
  detail?: string

  constructor(message: string, status: number, detail?: string) {
    super(message)
    this.name = "BackendProxyError"
    this.status = status
    this.detail = detail
  }
}

function backendConfig() {
  const localFallback = loadLocalBackendFallback()
  const isProduction = process.env.NODE_ENV === "production"

  const rawBaseUrl =
    process.env.PY_BACKEND_URL ||
    localFallback.PY_BACKEND_URL ||
    (!isProduction ? "http://localhost:8000" : undefined)
  const apiKey =
    process.env.PY_BACKEND_API_KEY ||
    localFallback.PY_BACKEND_API_KEY ||
    process.env.BACKEND_API_KEY ||
    localFallback.BACKEND_API_KEY

  const baseUrl = rawBaseUrl?.replace(/\/$/, "")
  const allowLocalDevFallback =
    !isProduction &&
    !!baseUrl &&
    isLoopbackBackendUrl(baseUrl)
  const resolvedApiKey = apiKey || (allowLocalDevFallback ? LOCAL_DEV_BACKEND_API_KEY : undefined)

  if (!baseUrl || !resolvedApiKey) {
    throw new BackendProxyError(
      "Python backend is not configured",
      503,
      "Set PY_BACKEND_URL and PY_BACKEND_API_KEY, or run the backend on localhost to use the built-in dev key fallback",
    )
  }

  return {
    baseUrl,
    apiKey: resolvedApiKey,
  }
}

type LocalBackendFallback = Partial<Record<"PY_BACKEND_URL" | "PY_BACKEND_API_KEY" | "BACKEND_API_KEY", string>>

let cachedLocalFallback: LocalBackendFallback | null = null

function loadLocalBackendFallback(): LocalBackendFallback {
  if (cachedLocalFallback) {
    return cachedLocalFallback
  }

  const cwd = process.cwd()
  const envCandidates = [
    path.resolve(cwd, ".env"),
    path.resolve(cwd, "..", ".env"),
  ]

  for (const envPath of envCandidates) {
    try {
      if (!fs.existsSync(envPath)) continue
      cachedLocalFallback = parseEnvFile(fs.readFileSync(envPath, "utf8"))
      return cachedLocalFallback
    } catch {
      continue
    }
  }

  cachedLocalFallback = {}
  return cachedLocalFallback
}

function parseEnvFile(contents: string): LocalBackendFallback {
  const parsed: LocalBackendFallback = {}

  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith("#")) continue

    const equalsIdx = line.indexOf("=")
    if (equalsIdx <= 0) continue

    const key = line.slice(0, equalsIdx).trim()
    const value = line.slice(equalsIdx + 1).trim().replace(/^['"]|['"]$/g, "")

    if (
      key === "PY_BACKEND_URL" ||
      key === "PY_BACKEND_API_KEY" ||
      key === "BACKEND_API_KEY"
    ) {
      parsed[key] = value
    }
  }

  return parsed
}

function isLoopbackBackendUrl(baseUrl: string): boolean {
  try {
    const { hostname } = new URL(baseUrl)
    return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1"
  } catch {
    return false
  }
}

export interface BackendRequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE"
  query?: Record<string, string | number | boolean | undefined | null>
  body?: unknown
  timeoutMs?: number
  headers?: Record<string, string>
}

export async function backendRequest<T>(
  path: string,
  options: BackendRequestOptions = {},
): Promise<T> {
  const { baseUrl, apiKey } = backendConfig()
  const method = options.method || "GET"
  const timeoutMs = options.timeoutMs ?? 30000

  const url = new URL(`${baseUrl}${path.startsWith("/") ? path : `/${path}`}`)
  if (options.query) {
    for (const [key, value] of Object.entries(options.query)) {
      if (value === undefined || value === null) continue
      url.searchParams.set(key, String(value))
    }
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
        "X-Backend-Api-Key": apiKey,
        ...(options.headers || {}),
      },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: controller.signal,
      cache: "no-store",
    })

    const contentType = response.headers.get("content-type") || ""
    const isJson = contentType.includes("application/json")
    const payload = isJson ? await response.json() : await response.text()

    if (!response.ok) {
      const detail =
        (typeof payload === "object" && payload && "detail" in payload
          ? String((payload as Record<string, unknown>).detail)
          : typeof payload === "object" && payload && "error" in payload
            ? String((payload as Record<string, unknown>).error)
            : undefined) || response.statusText

      throw new BackendProxyError(
        `Backend request failed: ${response.status}`,
        response.status,
        detail,
      )
    }

    return payload as T
  } catch (error) {
    if (error instanceof BackendProxyError) {
      throw error
    }

    if (error instanceof Error && error.name === "AbortError") {
      throw new BackendProxyError("Backend request timed out", 504, "Request exceeded timeout")
    }

    throw new BackendProxyError(
      "Backend unavailable",
      503,
      error instanceof Error ? error.message : "Unknown network error",
    )
  } finally {
    clearTimeout(timeout)
  }
}

export function backendErrorResponse(error: unknown, fallback = "Request failed") {
  if (error instanceof BackendProxyError) {
    const isAuthFailure = error.status === 401 || error.status === 403
    const mappedStatus = isAuthFailure ? 503 : error.status
    const mappedMessage = isAuthFailure ? "Backend service unavailable" : (error.detail || fallback)

    return NextResponse.json(
      {
        error: mappedMessage,
        code: mappedStatus,
      },
      { status: mappedStatus >= 400 && mappedStatus <= 599 ? mappedStatus : 500 },
    )
  }

  return NextResponse.json(
    {
      error: fallback,
      code: 500,
    },
    { status: 500 },
  )
}
