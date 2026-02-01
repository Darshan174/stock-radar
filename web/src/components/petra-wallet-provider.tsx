"use client"

/**
 * Petra Wallet Integration for x402 Payments
 * 
 * This provider enables users to connect their Petra wallet
 * for secure, user-friendly x402 payments.
 * 
 * ## Installation Required
 * 
 * ```bash
 * npm install @aptos-labs/wallet-adapter-react @aptos-labs/wallet-adapter-core
 * npm install @petra-wallet/aptos-wallet-adapter
 * ```
 * 
 * ## Features
 * 
 * - Connect Petra wallet with one click
 * - Sign x402 payment transactions
 * - View transaction history
 * - Gasless transaction support via facilitator
 */

import { createContext, useContext, useState, useCallback, ReactNode } from "react"

interface WalletContextType {
  account: string | null
  connected: boolean
  connecting: boolean
  connect: () => Promise<void>
  disconnect: () => void
  signAndSubmitTransaction: (transaction: any) => Promise<any>
  wallet: any
}

const WalletContext = createContext<WalletContextType | undefined>(undefined)

export function usePetraWallet() {
  const context = useContext(WalletContext)
  if (!context) {
    throw new Error("usePetraWallet must be used within PetraWalletProvider")
  }
  return context
}

interface PetraWalletProviderProps {
  children: ReactNode
}

export function PetraWalletProvider({ children }: PetraWalletProviderProps) {
  const [account, setAccount] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [wallet, setWallet] = useState<any>(null)

  // Check if Petra wallet is installed
  const isPetraInstalled = useCallback(() => {
    return typeof window !== "undefined" && (window as any).petra
  }, [])

  const connect = useCallback(async () => {
    if (!isPetraInstalled()) {
      window.open("https://petra.app/", "_blank")
      throw new Error("Petra wallet not installed. Please install it first.")
    }

    setConnecting(true)
    try {
      const petra = (window as any).petra
      
      // Request connection
      const response = await petra.connect()
      
      if (response.address) {
        setAccount(response.address)
        setConnected(true)
        setWallet(petra)
        console.log("Connected to Petra:", response.address)
      }
    } catch (error) {
      console.error("Failed to connect Petra:", error)
      throw error
    } finally {
      setConnecting(false)
    }
  }, [isPetraInstalled])

  const disconnect = useCallback(() => {
    if (wallet) {
      wallet.disconnect()
    }
    setAccount(null)
    setConnected(false)
    setWallet(null)
  }, [wallet])

  const signAndSubmitTransaction = useCallback(async (transaction: any) => {
    if (!wallet || !connected) {
      throw new Error("Wallet not connected")
    }

    try {
      // Sign and submit via Petra
      const pendingTxn = await wallet.signAndSubmitTransaction(transaction)
      return pendingTxn
    } catch (error) {
      console.error("Transaction failed:", error)
      throw error
    }
  }, [wallet, connected])

  return (
    <WalletContext.Provider
      value={{
        account,
        connected,
        connecting,
        connect,
        disconnect,
        signAndSubmitTransaction,
        wallet,
      }}
    >
      {children}
    </WalletContext.Provider>
  )
}

/**
 * Petra Wallet Button Component
 * 
 * One-click connect button for Petra wallet
 */
export function PetraConnectButton() {
  const { connected, connecting, connect, disconnect, account } = usePetraWallet()

  if (connected && account) {
    return (
      <button
        onClick={disconnect}
        className="flex items-center gap-2 px-4 py-2 bg-green-100 text-green-800 rounded-lg hover:bg-green-200 transition-colors"
      >
        <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
        <span className="font-mono text-sm">
          {account.slice(0, 6)}...{account.slice(-4)}
        </span>
      </button>
    )
  }

  return (
    <button
      onClick={connect}
      disabled={connecting}
      className="flex items-center gap-2 px-4 py-2 bg-teal-500 text-white rounded-lg hover:bg-teal-600 transition-colors disabled:opacity-50"
    >
      {connecting ? (
        <>
          <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          Connecting...
        </>
      ) : (
        <>
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
          </svg>
          Connect Petra
        </>
      )}
    </button>
  )
}

/**
 * Hook for x402 payments with Petra wallet
 * 
 * @example
 * ```typescript
 * const { payWithPetra } = usePetraPayment()
 * 
 * const txHash = await payWithPetra({
 *   recipient: "0x...",
 *   amount: 100,
 *   endpoint: "/api/agent/momentum"
 * })
 * ```
 */
export function usePetraPayment() {
  const { connected, signAndSubmitTransaction, account } = usePetraWallet()
  const [paying, setPaying] = useState(false)

  const payWithPetra = useCallback(async (params: {
    recipient: string
    amount: number
    endpoint: string
    gasless?: boolean
    facilitatorUrl?: string
  }): Promise<string> => {
    if (!connected) {
      throw new Error("Please connect Petra wallet first")
    }

    setPaying(true)
    try {
      const { recipient, amount, gasless } = params

      if (gasless) {
        // Gasless: Sign only, facilitator submits
        console.log("Using gasless mode with facilitator...")
        
        // Build transaction
        const transaction = {
          data: {
            function: "0x1::aptos_account::transfer",
            functionArguments: [recipient, amount],
          },
        }

        // Sign (don't submit)
        const signedTx = await signAndSubmitTransaction(transaction)
        
        // Send to facilitator
        const response = await fetch(`${params.facilitatorUrl || "https://x402-navy.vercel.app/facilitator/"}settle`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            signedTx: signedTx.hash,
            paymentRequest: {
              amount: amount.toString(),
              recipient,
              endpoint: params.endpoint,
            },
          }),
        })

        const result = await response.json()
        if (!result.success) {
          throw new Error(`Facilitator failed: ${result.error}`)
        }

        return result.txHash
      } else {
        // Direct: Sign and submit
        console.log("Submitting transaction directly...")
        
        const transaction = {
          data: {
            function: "0x1::aptos_account::transfer",
            functionArguments: [recipient, amount],
          },
        }

        const pendingTxn = await signAndSubmitTransaction(transaction)
        return pendingTxn.hash
      }
    } finally {
      setPaying(false)
    }
  }, [connected, signAndSubmitTransaction])

  return { payWithPetra, paying, connected, account }
}
