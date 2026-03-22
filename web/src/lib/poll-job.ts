import type { AnalyzeJobStatus } from "@/lib/analyze-contracts"

const INITIAL_DELAY_MS = 1000
const MAX_DELAY_MS = 5000
const BACKOFF_FACTOR = 2

export type PollProgressCallback = (status: AnalyzeJobStatus) => void

export async function pollAnalyzeJob(
  jobId: string,
  options?: {
    timeoutMs?: number
    onProgress?: PollProgressCallback
  },
): Promise<AnalyzeJobStatus> {
  const timeoutMs = options?.timeoutMs ?? 180_000
  const startedAt = Date.now()
  let delay = INITIAL_DELAY_MS

  while (Date.now() - startedAt < timeoutMs) {
    const res = await fetch(`/api/analyze/status?jobId=${encodeURIComponent(jobId)}`, {
      cache: "no-store",
    })

    // On 429 (rate limited) or 5xx, back off and retry instead of throwing
    if (res.status === 429 || res.status >= 500) {
      const retryAfter = res.headers.get("Retry-After")
      const retryMs = retryAfter ? parseInt(retryAfter, 10) * 1000 : delay * BACKOFF_FACTOR
      delay = Math.min(retryMs, MAX_DELAY_MS)
      await new Promise((resolve) => setTimeout(resolve, delay))
      continue
    }

    const data = (await res.json()) as AnalyzeJobStatus & { error?: string }

    if (!res.ok) {
      throw new Error(data.error || "Failed to fetch analysis status")
    }

    if (data.status === "succeeded") {
      return data
    }

    if (data.status === "failed") {
      throw new Error(data.error || "Analysis failed")
    }

    options?.onProgress?.(data)

    await new Promise((resolve) => setTimeout(resolve, delay))
    delay = Math.min(delay * BACKOFF_FACTOR, MAX_DELAY_MS)
  }

  throw new Error("Analysis timed out. Please try again.")
}
