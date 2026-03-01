import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseKey)

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
