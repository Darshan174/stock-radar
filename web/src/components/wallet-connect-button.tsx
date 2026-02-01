"use client"

/**
 * Wallet Connect Button for Petra
 * 
 * This uses the window.petra object directly for simple integration.
 * For production, use @aptos-labs/wallet-adapter-react
 */

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Loader2, Wallet, CheckCircle2 } from "lucide-react"

interface PetraWallet {
  connect: () => Promise<{ address: string }>
  disconnect: () => Promise<void>
  isConnected: () => Promise<boolean>
  account: () => Promise<{ address: string }>
  signAndSubmitTransaction: (transaction: any) => Promise<{ hash: string }>
}

export function WalletConnectButton() {
  const [connected, setConnected] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [address, setAddress] = useState<string | null>(null)
  const [wallet, setWallet] = useState<PetraWallet | null>(null)

  // Check if Petra is installed
  const checkPetra = useCallback(() => {
    if (typeof window !== "undefined") {
      const petra = (window as any).petra
      if (petra) {
        setWallet(petra)
        return true
      }
    }
    return false
  }, [])

  // Check connection on mount
  useEffect(() => {
    const init = async () => {
      if (checkPetra()) {
        try {
          const petra = (window as any).petra
          const isConnected = await petra.isConnected()
          if (isConnected) {
            const account = await petra.account()
            setAddress(account.address)
            setConnected(true)
          }
        } catch (error) {
          console.log("Not connected")
        }
      }
    }
    init()
  }, [checkPetra])

  const connect = async () => {
    if (!wallet) {
      // Open Petra installation page
      window.open("https://petra.app/", "_blank")
      alert("Please install Petra wallet extension and refresh the page")
      return
    }

    setConnecting(true)
    try {
      const response = await wallet.connect()
      setAddress(response.address)
      setConnected(true)
      console.log("Connected to Petra:", response.address)
    } catch (error) {
      console.error("Connection failed:", error)
      alert("Failed to connect Petra wallet. Please try again.")
    } finally {
      setConnecting(false)
    }
  }

  const disconnect = async () => {
    if (wallet) {
      try {
        await wallet.disconnect()
      } catch (e) {
        console.log("Disconnect error:", e)
      }
    }
    setAddress(null)
    setConnected(false)
  }

  // Check if Petra is installed
  const isPetraInstalled = !!wallet

  if (connected && address) {
    return (
      <Button
        variant="outline"
        onClick={disconnect}
        className="flex items-center gap-2 bg-green-50 border-green-200 hover:bg-green-100"
      >
        <CheckCircle2 className="h-4 w-4 text-green-600" />
        <span className="font-mono text-sm">
          {address.slice(0, 6)}...{address.slice(-4)}
        </span>
      </Button>
    )
  }

  return (
    <Button
      onClick={connect}
      disabled={connecting}
      className="flex items-center gap-2"
    >
      {connecting ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          Connecting...
        </>
      ) : (
        <>
          <Wallet className="h-4 w-4" />
          {isPetraInstalled ? "Connect Petra" : "Install Petra"}
        </>
      )}
    </Button>
  )
}

// Hook to use wallet in components
export function useWallet() {
  const [account, setAccount] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [wallet, setWallet] = useState<PetraWallet | null>(null)

  useEffect(() => {
    if (typeof window !== "undefined") {
      const petra = (window as any).petra
      if (petra) {
        setWallet(petra)
      }
    }
  }, [])

  useEffect(() => {
    const checkConnection = async () => {
      if (wallet) {
        try {
          const isConnected = await wallet.isConnected()
          if (isConnected) {
            const acc = await wallet.account()
            setAccount(acc.address)
            setConnected(true)
          }
        } catch (e) {
          console.log("Not connected")
        }
      }
    }
    checkConnection()
  }, [wallet])

  const signAndSubmitTransaction = async (transaction: any) => {
    if (!wallet || !connected) {
      throw new Error("Wallet not connected")
    }
    return wallet.signAndSubmitTransaction(transaction)
  }

  return {
    account,
    connected,
    wallet,
    signAndSubmitTransaction,
  }
}
