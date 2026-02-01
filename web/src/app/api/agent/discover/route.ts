import { NextRequest, NextResponse } from "next/server"
import { PROTECTED_ENDPOINTS } from "@/lib/x402-enforcer"

/**
 * Agent Discovery Endpoint
 * 
 * Allows other agents to discover this Financial Intelligence Agent's capabilities,
 * pricing, and contact information. Follows the Bazaar-style agent discovery pattern.
 * 
 * This endpoint is PUBLIC (no payment required) to enable agent discovery.
 * 
 * ## Bazaar Search Integration
 * 
 * This metadata is designed to be indexed by Bazaar-style search engines.
 * Agents can search for services by:
 * - Tags (financial-analysis, stock-analysis, etc.)
 * - Pricing range
 * - Capabilities
 * - Reputation score
 */

export async function GET(request: NextRequest) {
  const host = request.headers.get("host") || "localhost:3000"
  const protocol = host.includes("localhost") ? "http" : "https"
  const baseUrl = `${protocol}://${host}`

  // Build capabilities list from protected endpoints
  const capabilities = PROTECTED_ENDPOINTS.map(endpoint => {
    const name = endpoint.path.split("/").pop() || "unknown"
    return {
      name,
      endpoint: `${baseUrl}${endpoint.path}`,
      price: {
        amount: endpoint.price?.toString() || "100",
        currency: "APT",
        network: "testnet",
      },
      description: getCapabilityDescription(name),
      parameters: getCapabilityParameters(name),
    }
  })

  // Agent metadata with Bazaar search fields
  const agentMetadata = {
    // Agent Identity
    id: "stock-radar-financial-intelligence-v1",
    name: "Stock-Radar Financial Intelligence Agent",
    version: "1.0.0",
    type: "financial-analysis",
    
    // Bazaar Search Fields
    bazaar: {
      // Searchable tags
      tags: [
        "financial-intelligence",
        "stock-analysis",
        "technical-indicators",
        "ai-trading",
        "algorithmic-scoring",
        "momentum-trading",
        "value-investing",
        "sentiment-analysis",
        "aptos",
        "x402",
        "pay-per-use",
        "micro-payments",
        "gasless",
      ],
      // Price range for filtering
      pricingRange: {
        min: 50,      // octas
        max: 500,     // octas
        currency: "APT",
        model: "pay-per-use",
      },
      // Categories for browsing
      categories: [
        { name: "Financial Services", slug: "finance" },
        { name: "AI/ML", slug: "ai-ml" },
        { name: "Data Analysis", slug: "data" },
        { name: "Trading", slug: "trading" },
      ],
      // Supported markets/regions
      markets: ["US", "IN", "Global"],
      // Asset types supported
      assets: ["stocks", "etfs", "crypto"],
    },
    
    // Description
    description: "AI-powered stock analysis agent providing technical indicators, AI-based insights, and algorithmic scoring for financial markets. Pay-per-use with gasless transactions.",
    
    // Capabilities
    capabilities: {
      endpoints: capabilities,
      categories: [
        "technical-analysis",
        "fundamental-analysis",
        "sentiment-analysis",
        "algorithmic-scoring",
        "momentum-signals",
        "support-resistance",
      ],
    },
    
    // Pricing
    pricing: {
      currency: "APT",
      network: process.env.APTOS_NETWORK?.includes("mainnet") ? "mainnet" : "testnet",
      model: "pay-per-use",
      note: "Each API call requires a separate micropayment. No subscriptions or accounts needed. Gasless transactions available via facilitator.",
      gasless: {
        enabled: true,
        facilitator: process.env.FACILITATOR_URL || "https://x402-navy.vercel.app/facilitator/",
      },
    },
    
    // Payment Configuration
    payment: {
      recipientAddress: process.env.APTOS_RECIPIENT_ADDRESS || "0x1",
      requiredHeaders: ["X-Payment-Tx"],
      paymentInstructions: {
        step1: "Send APT payment to the recipient address",
        step2: "Include transaction hash in X-Payment-Tx header",
        step3: "Receive analysis response after payment verification",
        gasless: "Or use facilitator for gasless transactions - sign only, facilitator pays gas!",
      },
    },
    
    // Reputation (placeholder for on-chain tracking)
    reputation: {
      totalRequests: 0,      // TODO: Track on-chain
      successfulRequests: 0, // TODO: Track on-chain
      failedRequests: 0,     // TODO: Track on-chain
      averageRating: 0,      // TODO: Implement rating system
      totalVolume: "0",      // TODO: Track total APT volume
      // Bazaar reputation score
      bazaarScore: 0,
    },
    
    // Contact & Communication
    contact: {
      // XMTP-style messaging via HTTP
      xmtp: {
        enabled: true,
        address: null,
        messageEndpoint: `${baseUrl}/api/agent/message`,
        protocol: "agent-xmtp-v1",
        note: "Send agent-xmtp-v1 messages to the message endpoint for capability discovery, pricing inquiries, and task requests.",
      },
      
      // API Endpoints
      api: {
        baseUrl,
        discovery: `${baseUrl}/api/agent/discover`,
        health: `${baseUrl}/api/health`,
        demo: `${baseUrl}/x402-demo`,
        dashboard: `${baseUrl}/usage-dashboard`,
        message: `${baseUrl}/api/agent/message`, // Future: Direct agent messaging
      },
      
      // Documentation
      documentation: `${baseUrl}/docs`, // Future: API documentation
    },
    
    // Supported Markets
    markets: {
      regions: ["US", "IN", "Global"],
      assetTypes: ["stocks", "etfs", "crypto"],
      exchanges: ["NYSE", "NASDAQ", "NSE", "BSE"],
    },
    
    // AI Models Used
    aiModels: {
      primary: "groq/llama-3.3-70b-versatile",
      fallback: ["gemini/gemini-2.0-flash", "ollama/mistral"],
      scoringMethod: "algorithmic",
    },
    
    // Technical Specs
    specs: {
      responseFormat: "JSON",
      maxResponseTime: "30s",
      rateLimit: "No limit - pay per request",
      caching: "None - real-time data",
      verificationModes: ["direct", "facilitator", "hybrid"],
    },
    
    // On-Chain Registry
    onChain: {
      enabled: true,
      contractAddress: "0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240",
      modules: ["minimal_registry", "agent_registry", "agent_marketplace"],
      network: "aptos-devnet",
      explorer: "https://explorer.aptoslabs.com/account/0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240/modules?network=devnet",
    },
    
    // Timestamp
    discoveredAt: new Date().toISOString(),
  }

  return NextResponse.json(agentMetadata, {
    headers: {
      "Cache-Control": "public, max-age=60", // Cache for 1 minute
    },
  })
}

/**
 * Get human-readable description for each capability
 */
function getCapabilityDescription(name: string): string {
  const descriptions: Record<string, string> = {
    "momentum": "Calculate momentum signals using RSI, MACD, and price trends. Returns bullish/bearish/neutral signal with confidence score.",
    "rsi-divergence": "Detect RSI divergence patterns for potential trend reversals. Identifies bullish and bearish divergences.",
    "news-impact": "Analyze recent news sentiment and its potential impact on stock price. Includes sector context and key topics.",
    "stock-score": "Comprehensive algorithmic scoring across momentum, value, quality, and risk dimensions. Overall recommendation included.",
    "social-sentiment": "Fetch social media sentiment from Reddit and other platforms. Includes mention counts and ranking.",
    "support-resistance": "Calculate support and resistance levels using pivot points, Bollinger Bands, and ATR.",
    "analyze": "Full AI-powered stock analysis with intraday or long-term trading recommendations. Most comprehensive endpoint.",
    "fundamentals": "Retrieve company fundamentals including P/E, P/B, ROE, margins, debt ratios, and analyst targets.",
    "orchestrate": "Comprehensive multi-signal analysis combining momentum (40%), fundamentals (40%), and sentiment (20%). Runs all sub-analyses in parallel and returns a weighted aggregate score with overall signal. Discounted vs calling individual endpoints.",
    "live-price": "Get real-time stock price, change, volume, and OHLC data with 5-second cache.",
  }
  
  return descriptions[name] || "Financial analysis endpoint"
}

/**
 * Get expected parameters for each capability
 */
function getCapabilityParameters(name: string): Record<string, any> {
  const commonParams = {
    symbol: {
      type: "string",
      required: true,
      description: "Stock symbol (e.g., AAPL, RELIANCE.NS, TSLA)",
      example: "AAPL",
    },
  }

  const specificParams: Record<string, Record<string, any>> = {
    "momentum": {
      period: {
        type: "string",
        required: false,
        default: "14",
        description: "RSI period for momentum calculation",
      },
    },
    "rsi-divergence": {
      period: {
        type: "string",
        required: false,
        default: "14",
        description: "RSI calculation period",
      },
      lookback: {
        type: "string",
        required: false,
        default: "5",
        description: "Days to look back for divergence detection",
      },
    },
    "news-impact": {
      days: {
        type: "string",
        required: false,
        default: "7",
        description: "Number of days of news to analyze",
      },
    },
    "support-resistance": {
      period: {
        type: "string",
        required: false,
        default: "14",
        description: "Period for calculations",
      },
    },
    "analyze": {
      _method: {
        type: "string",
        value: "POST",
        description: "This endpoint requires a POST request with a JSON body",
      },
      mode: {
        type: "string",
        required: false,
        default: "intraday",
        description: "Analysis mode: intraday or longterm (JSON body)",
        enum: ["intraday", "longterm"],
      },
      period: {
        type: "string",
        required: false,
        default: "max",
        description: "Historical data period (JSON body)",
        enum: ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
      },
    },
    "orchestrate": {
      type: {
        type: "string",
        required: false,
        default: "comprehensive-analysis",
        description: "Orchestration type",
      },
    },
  }

  return {
    ...commonParams,
    ...(specificParams[name] || {}),
  }
}
