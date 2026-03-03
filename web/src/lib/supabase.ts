import { createClient, type SupabaseClient } from '@supabase/supabase-js'

const rawSupabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const rawSupabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
const supabaseUrl = rawSupabaseUrl?.trim()
const supabaseKey = rawSupabaseKey?.trim()
const hasSupabaseEnv = Boolean(supabaseUrl && supabaseKey)

if (!hasSupabaseEnv && typeof window !== 'undefined') {
    console.error(
        "Missing Supabase env vars. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in Vercel."
    )
}

let supabaseClient: SupabaseClient | null = null

function getSupabaseClient(): SupabaseClient {
    if (supabaseClient) return supabaseClient

    // Keep app booting even when envs are missing so we can render a clear UI error instead of crashing.
    const resolvedUrl = supabaseUrl || 'https://example.supabase.co'
    const resolvedKey = supabaseKey || 'public-anon-key-placeholder'

    supabaseClient = createClient(resolvedUrl, resolvedKey)
    return supabaseClient
}

// Lazy proxy prevents createClient() from running during SSR module evaluation.
export const supabase: SupabaseClient = new Proxy({} as SupabaseClient, {
    get(_target, prop, receiver) {
        const client = getSupabaseClient()
        const value = Reflect.get(client, prop, receiver)
        return typeof value === 'function' ? value.bind(client) : value
    },
})
export { hasSupabaseEnv }

// Database types
export interface Stock {
    id: number
    symbol: string
    name: string
    exchange: string
    sector: string | null
    currency: string
    created_at: string
}

export interface Analysis {
    id: number
    stock_id: number
    mode: 'intraday' | 'longterm'
    signal: string
    confidence: number
    reasoning: string
    technical_summary: string | null
    sentiment_summary: string | null
    support_level: number | null
    resistance_level: number | null
    target_price: number | null
    stop_loss: number | null
    llm_model: string
    embedding_text?: string | null
    created_at: string
    stocks?: Stock
}

export interface Signal {
    id: number
    stock_id: number
    signal_type: string
    signal: string
    price: number
    reason: string
    importance: string
    created_at: string
    stocks?: Stock
}

export interface PriceHistory {
    id: number
    stock_id: number
    timestamp: string
    open: number
    high: number
    low: number
    close: number
    volume: number
}
