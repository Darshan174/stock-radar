import { x402Enforcer } from "./x402-enforcer"

export function validateX402Config(): { 
  valid: boolean
  errors: string[]
  warnings: string[]
  info: string[]
} {
  const errors: string[] = []
  const warnings: string[] = []
  const info: string[] = []

  const validation = x402Enforcer.validateConfig()
  errors.push(...validation.errors)

  // Check recipient address
  if (!process.env.APTOS_RECIPIENT_ADDRESS) {
    errors.push("APTOS_RECIPIENT_ADDRESS environment variable is missing")
  } else if (process.env.APTOS_RECIPIENT_ADDRESS === "0x1") {
    warnings.push("APTOS_RECIPIENT_ADDRESS is set to default (0x1). This is a placeholder and should be replaced with your actual Aptos wallet address.")
  } else {
    info.push(`Payment recipient: ${process.env.APTOS_RECIPIENT_ADDRESS}`)
  }

  // Check price per request
  if (!process.env.APTOS_PRICE_PER_REQUEST) {
    warnings.push("APTOS_PRICE_PER_REQUEST is not set. Using default: 100 octas (0.000001 APT)")
  } else {
    const price = parseInt(process.env.APTOS_PRICE_PER_REQUEST)
    if (isNaN(price) || price <= 0) {
      errors.push("APTOS_PRICE_PER_REQUEST must be a positive integer")
    } else {
      const aptAmount = price / 100000000
      info.push(`Default price per request: ${price} octas (${aptAmount.toFixed(8)} APT)`)
    }
  }

  // Check network configuration
  if (!process.env.APTOS_NETWORK) {
    warnings.push("APTOS_NETWORK not set. Using default: https://fullnode.testnet.aptoslabs.com/v1")
  } else {
    const network = process.env.APTOS_NETWORK
    if (network.includes("mainnet")) {
      warnings.push("⚠️  Using MAINNET - Real APT will be required for payments")
    } else if (network.includes("testnet")) {
      info.push("Using Aptos Testnet")
    } else if (network.includes("devnet")) {
      info.push("Using Aptos Devnet")
    } else if (network.includes("localhost")) {
      info.push("Using local Aptos node")
    }
  }

  // Check Supabase configuration (for data storage)
  if (!process.env.NEXT_PUBLIC_SUPABASE_URL) {
    warnings.push("NEXT_PUBLIC_SUPABASE_URL not set. Some features may not work.")
  }
  if (!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY) {
    warnings.push("NEXT_PUBLIC_SUPABASE_ANON_KEY not set. Database features disabled.")
  }

  // Check AI model API keys
  const hasZai = !!process.env.ZAI_API_KEY
  const hasGemini = !!process.env.GEMINI_API_KEY

  if (!hasZai && !hasGemini) {
    warnings.push("No AI API keys configured (ZAI_API_KEY or GEMINI_API_KEY). AI analysis will fall back to local Ollama.")
  } else {
    const providers = []
    if (hasZai) providers.push("ZAI (GLM-4.7)")
    if (hasGemini) providers.push("Gemini")
    info.push(`AI providers available: ${providers.join(", ")}`)
  }

  // Summary
  if (errors.length === 0 && warnings.length === 0) {
    console.log("✓ X402 Payment Configuration Valid")
    console.log(`  Recipient: ${x402Enforcer.getRecipientAddress()}`)
    console.log(`  Network: ${process.env.APTOS_NETWORK || "testnet (default)"}`)
    console.log(`  Default price: ${x402Enforcer.getPricePerRequest()} octas`)
  }

  return { valid: errors.length === 0, errors, warnings, info }
}

export function startupX402Validation(): void {
  const validation = validateX402Config()

  console.log("\n" + "=".repeat(60))
  console.log("X402 PAYMENT SYSTEM - STARTUP VALIDATION")
  console.log("=".repeat(60))

  // Info messages
  if (validation.info.length > 0) {
    console.log("\nℹ️  Configuration:")
    validation.info.forEach(msg => console.log(`  • ${msg}`))
  }

  // Warnings
  if (validation.warnings.length > 0) {
    console.warn("\n⚠️  Warnings:")
    validation.warnings.forEach(warning => console.warn(`  • ${warning}`))
  }

  // Errors
  if (validation.errors.length > 0) {
    console.error("\n❌ Errors:")
    validation.errors.forEach(error => console.error(`  • ${error}`))
    console.error("\n⚠️  Please fix these errors before deploying to production.")
    console.error("   See: https://github.com/aptos-labs/aptos-x402 for setup instructions.\n")
  } else {
    console.log("\n✅ X402 configuration validated successfully!\n")
  }
}

/**
 * Get current X402 status for health checks
 */
export function getX402Status(): {
  configured: boolean
  network: string
  recipient: string
  defaultPrice: number
  endpoints: number
} {
  return {
    configured: x402Enforcer.isConfigured(),
    network: process.env.APTOS_NETWORK || "https://fullnode.testnet.aptoslabs.com/v1",
    recipient: x402Enforcer.getRecipientAddress(),
    defaultPrice: x402Enforcer.getPricePerRequest(),
    endpoints: 9, // Total number of protected endpoints
  }
}
