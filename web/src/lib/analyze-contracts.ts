export type AnalyzeJobState = "queued" | "running" | "succeeded" | "failed"

export interface AnalyzeJobCreated {
  jobId: string
  statusUrl: string
  status: AnalyzeJobState
}

export interface AnalyzeJobStatus {
  jobId: string
  status: AnalyzeJobState
  result?: Record<string, unknown>
  error?: string
}
