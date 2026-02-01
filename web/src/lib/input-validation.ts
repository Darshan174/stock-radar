/**
 * Input validation for API endpoints
 *
 * Prevents command injection when passing user input to child processes.
 */

/** Validates stock symbol - only allows alphanumeric, dots, hyphens, carets */
const SYMBOL_REGEX = /^[A-Za-z0-9.\-^]{1,20}$/

/** Validates numeric string parameters */
const NUMERIC_REGEX = /^[0-9]{1,5}$/

export function validateSymbol(symbol: string | null): { valid: true; value: string } | { valid: false; error: string } {
  if (!symbol) {
    return { valid: false, error: "Symbol is required" }
  }
  if (!SYMBOL_REGEX.test(symbol)) {
    return { valid: false, error: "Invalid symbol format. Only letters, numbers, dots, hyphens, and carets are allowed (max 20 chars)." }
  }
  return { valid: true, value: symbol.toUpperCase() }
}

export function validateNumericParam(
  value: string | null,
  defaultValue: string,
  name: string,
  min: number = 1,
  max: number = 365
): { valid: true; value: number } | { valid: false; error: string } {
  const raw = value || defaultValue
  if (!NUMERIC_REGEX.test(raw)) {
    return { valid: false, error: `Invalid ${name}: must be a number` }
  }
  const num = parseInt(raw, 10)
  if (num < min || num > max) {
    return { valid: false, error: `Invalid ${name}: must be between ${min} and ${max}` }
  }
  return { valid: true, value: num }
}
