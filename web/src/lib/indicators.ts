/**
 * Technical Indicators Library
 * Calculations for stock chart overlays and panels
 */

export interface OHLCData {
    time: string
    open: number
    high: number
    low: number
    close: number
    volume?: number
}

// Simple Moving Average
export function calculateSMA(data: number[], period: number): (number | null)[] {
    const sma: (number | null)[] = []

    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            sma.push(null)
        } else {
            const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0)
            sma.push(sum / period)
        }
    }

    return sma
}

// Exponential Moving Average
export function calculateEMA(data: number[], period: number): (number | null)[] {
    const ema: (number | null)[] = []
    const multiplier = 2 / (period + 1)

    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            ema.push(null)
        } else if (i === period - 1) {
            const sum = data.slice(0, period).reduce((a, b) => a + b, 0)
            ema.push(sum / period)
        } else {
            const prevEma = ema[i - 1]
            if (prevEma !== null) {
                ema.push((data[i] - prevEma) * multiplier + prevEma)
            } else {
                ema.push(null)
            }
        }
    }
    return ema
}

// Bollinger Bands
export interface BollingerBands {
    upper: (number | null)[]
    middle: (number | null)[]  // SMA
    lower: (number | null)[]
}

export function calculateBollingerBands(
    data: number[],
    period: number = 20,
    stdDevMultiplier: number = 2
): BollingerBands {
    const middle = calculateSMA(data, period)
    const upper: (number | null)[] = []
    const lower: (number | null)[] = []

    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            upper.push(null)
            lower.push(null)
        } else {
            const slice = data.slice(i - period + 1, i + 1)
            const mean = middle[i]!
            const squaredDiffs = slice.map(x => Math.pow(x - mean, 2))
            const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period
            const stdDev = Math.sqrt(variance)

            upper.push(mean + stdDevMultiplier * stdDev)
            lower.push(mean - stdDevMultiplier * stdDev)
        }
    }

    return { upper, middle, lower }
}

// RSI (Relative Strength Index)
export function calculateRSI(data: number[], period: number = 14): (number | null)[] {
    const rsi: (number | null)[] = []
    const gains: number[] = []
    const losses: number[] = []

    // Calculate price changes
    for (let i = 1; i < data.length; i++) {
        const change = data[i] - data[i - 1]
        gains.push(change > 0 ? change : 0)
        losses.push(change < 0 ? Math.abs(change) : 0)
    }

    // First RSI value (null for first period)
    rsi.push(null) // No change for first data point

    for (let i = 0; i < gains.length; i++) {
        if (i < period - 1) {
            rsi.push(null)
        } else if (i === period - 1) {
            // First RSI: simple average
            const avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period
            const avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period

            if (avgLoss === 0) {
                rsi.push(100)
            } else {
                const rs = avgGain / avgLoss
                rsi.push(100 - (100 / (1 + rs)))
            }
        } else {
            // Subsequent RSI: smoothed average
            const prevRsi = rsi[i]
            if (prevRsi === null) {
                rsi.push(null)
                continue
            }

            // Get previous averages from RS formula
            const prevGainSlice = gains.slice(i - period, i)
            const prevLossSlice = losses.slice(i - period, i)
            const prevAvgGain = prevGainSlice.reduce((a, b) => a + b, 0) / period
            const prevAvgLoss = prevLossSlice.reduce((a, b) => a + b, 0) / period

            // Smoothed averages
            const avgGain = (prevAvgGain * (period - 1) + gains[i]) / period
            const avgLoss = (prevAvgLoss * (period - 1) + losses[i]) / period

            if (avgLoss === 0) {
                rsi.push(100)
            } else {
                const rs = avgGain / avgLoss
                rsi.push(100 - (100 / (1 + rs)))
            }
        }
    }

    return rsi
}

// MACD (Moving Average Convergence Divergence)
export interface MACDResult {
    macdLine: (number | null)[]
    signalLine: (number | null)[]
    histogram: (number | null)[]
}

export function calculateMACD(
    data: number[],
    fastPeriod: number = 12,
    slowPeriod: number = 26,
    signalPeriod: number = 9
): MACDResult {
    const fastEMA = calculateEMA(data, fastPeriod)
    const slowEMA = calculateEMA(data, slowPeriod)

    // MACD Line = Fast EMA - Slow EMA
    const macdLine: (number | null)[] = fastEMA.map((fast, i) => {
        const slow = slowEMA[i]
        if (fast === null || slow === null) return null
        return fast - slow
    })

    // Signal Line = EMA of MACD Line
    const macdValues = macdLine.filter((v): v is number => v !== null)
    const signalEMA = calculateEMA(macdValues, signalPeriod)

    // Map signal EMA back to full length array
    const signalLine: (number | null)[] = []
    let signalIdx = 0
    for (let i = 0; i < macdLine.length; i++) {
        if (macdLine[i] === null) {
            signalLine.push(null)
        } else {
            signalLine.push(signalEMA[signalIdx] ?? null)
            signalIdx++
        }
    }

    // Histogram = MACD Line - Signal Line
    const histogram: (number | null)[] = macdLine.map((macd, i) => {
        const signal = signalLine[i]
        if (macd === null || signal === null) return null
        return macd - signal
    })

    return { macdLine, signalLine, histogram }
}

// Volume SMA
export function calculateVolumeSMA(volumes: number[], period: number = 20): (number | null)[] {
    return calculateSMA(volumes, period)
}

// Average True Range (ATR) - useful for volatility
export function calculateATR(data: OHLCData[], period: number = 14): (number | null)[] {
    const trueRanges: number[] = []

    for (let i = 0; i < data.length; i++) {
        if (i === 0) {
            trueRanges.push(data[i].high - data[i].low)
        } else {
            const high = data[i].high
            const low = data[i].low
            const prevClose = data[i - 1].close

            const tr = Math.max(
                high - low,
                Math.abs(high - prevClose),
                Math.abs(low - prevClose)
            )
            trueRanges.push(tr)
        }
    }

    return calculateSMA(trueRanges, period)
}

// Support and Resistance levels (simple pivot points)
export interface PivotPoints {
    pivot: number
    r1: number
    r2: number
    r3: number
    s1: number
    s2: number
    s3: number
}

export function calculatePivotPoints(high: number, low: number, close: number): PivotPoints {
    const pivot = (high + low + close) / 3

    return {
        pivot,
        r1: 2 * pivot - low,
        r2: pivot + (high - low),
        r3: high + 2 * (pivot - low),
        s1: 2 * pivot - high,
        s2: pivot - (high - low),
        s3: low - 2 * (high - pivot),
    }
}
