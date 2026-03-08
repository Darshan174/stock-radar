export type AnalysisMode = "intraday" | "longterm"

interface StoredSettings {
  defaultMode?: AnalysisMode
}

export interface AnalysisModeMeta {
  value: AnalysisMode
  label: string
  description: string
  defaultPeriod: string
}

export const ANALYSIS_MODE_META: Record<AnalysisMode, AnalysisModeMeta> = {
  intraday: {
    value: "intraday",
    label: "Intraday",
    description: "Short-horizon setup using recent price action, volume, momentum, and fresh news flow.",
    defaultPeriod: "5d",
  },
  longterm: {
    value: "longterm",
    label: "Long-term",
    description: "Fundamental and valuation view using weekly trend context and multi-quarter business quality.",
    defaultPeriod: "5y",
  },
}

export const ANALYSIS_MODE_OPTIONS = [
  ANALYSIS_MODE_META.intraday,
  ANALYSIS_MODE_META.longterm,
] as const

export function parseAnalysisMode(value: unknown): AnalysisMode {
  return value === "longterm" ? "longterm" : "intraday"
}

export function getAnalysisModeLabel(mode: AnalysisMode): string {
  return ANALYSIS_MODE_META[mode].label
}

export function getAnalysisModeDescription(mode: AnalysisMode): string {
  return ANALYSIS_MODE_META[mode].description
}

export function getDefaultAnalysisPeriod(mode: AnalysisMode): string {
  return ANALYSIS_MODE_META[mode].defaultPeriod
}

export function loadStoredAnalysisMode(): AnalysisMode {
  if (typeof window === "undefined") return "intraday"

  try {
    const rawSettings = window.localStorage.getItem("stock-radar-settings")
    if (!rawSettings) return "intraday"

    const parsed = JSON.parse(rawSettings) as StoredSettings
    return parseAnalysisMode(parsed.defaultMode)
  } catch {
    return "intraday"
  }
}
