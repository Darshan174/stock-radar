"use client"

import { useState, useEffect } from "react"
import { useWallet } from "@aptos-labs/wallet-adapter-react"
import { x402Client, X402Client } from "@/lib/x402-client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"

import {
  Loader2,
  Wallet,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  Zap,
  Code2,
  Database,
  CircleDot,
  Ban,
  Send,
  ShieldCheck,
  FileJson,
  Activity,
  TrendingUp,

  RefreshCw,
} from "lucide-react"

const ENDPOINTS = [
  { value: "/api/agent/momentum", label: "Momentum Analysis", price: "100 octas", method: "GET" as const },
  { value: "/api/agent/stock-score", label: "Stock Score", price: "200 octas", method: "GET" as const },
  { value: "/api/agent/support-resistance", label: "Support/Resistance", price: "100 octas", method: "GET" as const },
  { value: "/api/agent/news-impact", label: "News Impact", price: "150 octas", method: "GET" as const },
  { value: "/api/agent/social-sentiment", label: "Social Sentiment", price: "100 octas", method: "GET" as const },
  { value: "/api/agent/rsi-divergence", label: "RSI Divergence", price: "100 octas", method: "GET" as const },
  { value: "/api/agent/orchestrate", label: "Full Analysis (Orchestrate)", price: "400 octas", method: "POST" as const },
  { value: "/api/fundamentals", label: "Fundamentals", price: "100 octas", method: "GET" as const },
  { value: "/api/live-price", label: "Live Price", price: "50 octas", method: "GET" as const },
]

// Test wallet addresses — paste the corresponding private key from the hackathon cheatsheet.
const TEST_WALLETS = [
  { name: "Wallet 1", address: "0xaaefee8ba1e5f24ef88a74a3f445e0d2b810b90c1996466dae5ea9a0b85d42a0" },
  { name: "Wallet 2", address: "0xaaea48900c8f8045876505fe5fc5a623b1e423ef573a55b8b308cdecc749e6f4" },
  { name: "Wallet 3", address: "0x924c2e983753bb29b45ae9b4036d48861f204da096b36af710c95d1742b05ad4" },
  { name: "Wallet 4", address: "0xf1697d22257fd39653319eb3a2ee23fca2ca99b26f7fc79090249fbfbc401e03" },
  { name: "Wallet 5", address: "0x6cd199bbbc8bb3c17de4d2aebc2e75b4e9d7e3083188d987b597a3de8239df2a" },
]

// ── Milestone tracker for the flow visualization ──
type Milestone = "idle" | "requesting" | "402_received" | "payment_sent" | "payment_verified" | "data_returned" | "error"

const MILESTONES: { key: Milestone; label: string; icon: typeof CircleDot }[] = [
  { key: "402_received", label: "402 Received", icon: Ban },
  { key: "payment_sent", label: "Payment Sent", icon: Send },
  { key: "payment_verified", label: "Verified", icon: ShieldCheck },
  { key: "data_returned", label: "Data Returned", icon: FileJson },
]

function MilestoneBar({ current }: { current: Milestone }) {
  const order: Milestone[] = ["402_received", "payment_sent", "payment_verified", "data_returned"]
  const currentIdx = order.indexOf(current)

  return (
    <div className="flex items-center gap-1 w-full">
      {MILESTONES.map((m, i) => {
        const done = currentIdx >= i
        const active = currentIdx === i
        const Icon = m.icon
        return (
          <div key={m.key} className="flex items-center flex-1 min-w-0">
            <div className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium whitespace-nowrap transition-colors ${done
              ? "bg-teal-500/15 text-teal-400 border border-teal-500/30"
              : "bg-muted text-muted-foreground border border-transparent"
              } ${active ? "ring-1 ring-teal-500/50" : ""}`}>
              <Icon className="h-3 w-3 flex-shrink-0" />
              <span className="hidden sm:inline">{m.label}</span>
            </div>
            {i < MILESTONES.length - 1 && (
              <div className={`flex-1 h-px mx-1 ${currentIdx > i ? "bg-teal-500/40" : "bg-border"}`} />
            )}
          </div>
        )
      })}
    </div>
  )
}

export default function X402DemoPage() {
  // Wallet adapter hook (for Petra / any AIP-62 wallet)
  const {
    connect: walletConnect,
    disconnect: walletDisconnect,
    account,
    connected: walletConnected,
    signAndSubmitTransaction,
    wallets,
  } = useWallet()

  // State
  const [privateKey, setPrivateKey] = useState("")
  const [usePetraWallet, setUsePetraWallet] = useState(false)
  const [client, setClient] = useState<X402Client | null>(null)
  const [walletAddress, setWalletAddress] = useState<string | null>(null)
  const [balance, setBalance] = useState<string | null>(null)
  const [gasless, setGasless] = useState(false)
  const [loading, setLoading] = useState(false)
  const [connecting, setConnecting] = useState(false)

  // API call state
  const [selectedEndpoint, setSelectedEndpoint] = useState("/api/agent/momentum")
  const [symbol, setSymbol] = useState("AAPL")
  const [result, setResult] = useState<any>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [txHash, setTxHash] = useState<string | null>(null)
  const [milestone, setMilestone] = useState<Milestone>("idle")
  const [reputation, setReputation] = useState<any>(null)
  const [reputationLoading, setReputationLoading] = useState(false)

  const fetchReputation = async () => {
    setReputationLoading(true)
    try {
      const res = await fetch("/api/agent/reputation")
      if (res.ok) {
        const data = await res.json()
        setReputation(data)
      }
    } catch {
      // silently ignore
    } finally {
      setReputationLoading(false)
    }
  }

  // Fetch reputation on mount
  useEffect(() => {
    fetchReputation()
  }, [])

  const addLog = (message: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`])
  }

  // ── Private Key Connect ──
  const handleConnectPrivateKey = async () => {
    if (!privateKey) {
      setError("Enter a private key or click a test wallet below")
      return
    }

    setConnecting(true)
    setError(null)
    setLogs([])

    try {
      addLog("Initializing x402 client...")

      const newClient = x402Client({
        privateKey,
        network: "testnet",
        gasless,
      })

      const address = newClient.getAddress()
      setWalletAddress(address)
      addLog(`Connected: ${address.slice(0, 8)}...${address.slice(-6)}`)

      try {
        const bal = await newClient.getBalance()
        setBalance(bal)
        addLog(`Balance: ${(Number(bal) / 100000000).toFixed(4)} APT (${bal} octas)`)
      } catch {
        setBalance("0")
        addLog("Balance: 0 APT — wallet needs testnet funds to make payments.")
      }

      setClient(newClient)
      addLog("Wallet ready — pick an endpoint and call the API.")

    } catch (err: any) {
      const raw = err.message || ""
      let msg: string
      if (raw.includes("endsWith") || raw.includes("privateKey") || raw.includes("Hex")) {
        msg = "Invalid private key format. It should start with 0x."
      } else if (raw.includes("Account not found") || raw.includes("resource_not_found")) {
        msg = "Wallet not found on testnet. Fund it at aptos.dev/network/faucet first."
      } else {
        msg = "Could not connect wallet. Check the private key and try again."
      }
      setError(msg)
      addLog(msg)
    } finally {
      setConnecting(false)
    }
  }

  // ── Petra / Wallet Adapter Connect ──
  const handleConnectPetra = async () => {
    setConnecting(true)
    setError(null)
    setLogs([])

    try {
      addLog("Connecting via Wallet Adapter...")

      const petraWallet = wallets.find(w => w.name === "Petra")
      if (!petraWallet) {
        throw new Error("Petra wallet not detected. Install the Petra browser extension.")
      }

      walletConnect(petraWallet.name)
      addLog("Waiting for Petra approval...")

    } catch (err: any) {
      setError(`Wallet connection failed: ${err.message}`)
      addLog(`Error: ${err.message}`)
      setConnecting(false)
    }
  }

  // Sync wallet adapter state
  const petraAddress = usePetraWallet && walletConnected && account?.address
    ? account.address.toString()
    : null

  useEffect(() => {
    if (connecting && petraAddress) {
      setConnecting(false)
      setWalletAddress(petraAddress)
      setLogs(prev => [
        ...prev,
        `[${new Date().toLocaleTimeString()}] Petra connected: ${petraAddress.slice(0, 8)}...${petraAddress.slice(-6)}`,
        `[${new Date().toLocaleTimeString()}] Ready — Petra will prompt you to approve each payment.`,
      ])
    }
  }, [connecting, petraAddress])

  // ── Petra API Call ──
  const handleCallApiPetra = async () => {
    if (!petraAddress) {
      setError("Petra wallet not connected")
      return
    }

    setLoading(true)
    setResult(null)
    setError(null)
    setTxHash(null)
    setLogs([])
    setMilestone("requesting")

    try {
      const ep = ENDPOINTS.find(e => e.value === selectedEndpoint)
      const isPost = ep?.method === "POST"
      const url = isPost ? selectedEndpoint : `${selectedEndpoint}?symbol=${symbol}`
      const fetchOpts: RequestInit = isPost
        ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ symbol }) }
        : {}
      addLog(`Calling ${selectedEndpoint} for ${symbol}...`)

      const initialResponse = await fetch(url, fetchOpts)

      if (initialResponse.status !== 402) {
        const data = await initialResponse.json()
        setResult(data)
        setMilestone("data_returned")
        addLog("Endpoint did not require payment.")
        return
      }

      // 402 received
      const errorData = await initialResponse.json()
      const payment = errorData.payment
      setMilestone("402_received")
      addLog(`Server requires ${payment.amount} octas to proceed.`)

      // Payment
      addLog("Requesting Petra signature...")
      const txResult = await signAndSubmitTransaction({
        data: {
          function: "0x1::aptos_account::transfer",
          functionArguments: [payment.recipient, Number(payment.amount)],
        },
      })

      const hash = typeof txResult === 'object' && 'hash' in txResult
        ? (txResult as any).hash
        : String(txResult)

      setMilestone("payment_sent")
      addLog(`Payment sent. Tx: ${hash.slice(0, 12)}...`)
      setTxHash(hash)

      // Retry
      addLog("Retrying with payment proof...")
      const paidFetchOpts: RequestInit = isPost
        ? { method: "POST", headers: { "Content-Type": "application/json", "X-Payment-Tx": hash }, body: JSON.stringify({ symbol }) }
        : { headers: { "X-Payment-Tx": hash } }
      const paidResponse = await fetch(url, paidFetchOpts)

      if (!paidResponse.ok) {
        const errText = await paidResponse.text()
        throw new Error(`Server rejected: ${paidResponse.status}`)
      }

      setMilestone("payment_verified")
      addLog("Payment verified on-chain.")

      const data = await paidResponse.json()
      setMilestone("data_returned")
      addLog("Analysis data received.")
      setResult(data)

      // Refresh on-chain reputation (server records it after verified payment)
      setTimeout(() => fetchReputation(), 3000)

    } catch (err: any) {
      setMilestone("error")
      const raw = err.message || ""
      let msg: string
      if (raw.includes("User rejected") || raw.includes("user rejected")) {
        msg = "You declined the transaction in Petra."
      } else if (raw.includes("Insufficient") || raw.includes("insufficient")) {
        msg = "Insufficient balance. Fund your wallet at aptos.dev/network/faucet."
      } else if (raw.includes("resource_not_found") || raw.includes("Account not found")) {
        msg = "Wallet has no funds on testnet. Fund it at aptos.dev/network/faucet first."
      } else if (raw.includes("timeout") || raw.includes("TIMEOUT")) {
        msg = "Request timed out. The server may be starting up — try again."
      } else if (raw.includes("Server rejected") || raw.includes("402")) {
        msg = "Payment was not accepted. The transaction may have already been used."
      } else if (raw.includes("Failed to fetch") || raw.includes("NetworkError")) {
        msg = "Could not reach the server. Make sure it's running."
      } else {
        msg = "Something went wrong. Check the console for details."
        console.error("x402 Petra error:", raw)
      }
      setError(msg)
      addLog(msg)
    } finally {
      setLoading(false)
    }
  }

  // ── Private Key API Call ──
  const handleCallApiPrivateKey = async () => {
    if (!client) {
      setError("Connect a wallet first")
      return
    }

    setLoading(true)
    setResult(null)
    setError(null)
    setTxHash(null)
    setLogs([])
    setMilestone("requesting")

    try {
      const ep = ENDPOINTS.find(e => e.value === selectedEndpoint)
      const isPost = ep?.method === "POST"
      const url = isPost ? selectedEndpoint : `${selectedEndpoint}?symbol=${symbol}`
      addLog(`Calling ${selectedEndpoint} for ${symbol}...`)
      addLog(gasless ? "Using gasless mode (facilitator pays gas)." : "Paying gas directly.")
      addLog("SDK handling 402 → payment → retry automatically...")

      const data = isPost
        ? await client.post(url, { symbol })
        : await client.get(url)

      // SDK completed the full flow: 402 → pay → verify → data
      setMilestone("data_returned")
      addLog("402 received, payment signed & submitted, verified, data returned.")
      setResult(data)

      if (data.txHash) {
        setTxHash(data.txHash)
      }

      // Refresh on-chain reputation
      setTimeout(() => fetchReputation(), 3000)

    } catch (err: any) {
      setMilestone("error")
      const raw = err.message || ""
      let msg: string
      if (raw.includes("Insufficient") || raw.includes("insufficient")) {
        msg = "Insufficient balance. Fund the wallet at aptos.dev/network/faucet."
      } else if (raw.includes("resource_not_found") || raw.includes("Account not found")) {
        msg = "Wallet has no funds on testnet. Fund it at aptos.dev/network/faucet first."
      } else if (raw.includes("timeout") || raw.includes("TIMEOUT")) {
        msg = "Request timed out. The server may be starting up — try again."
      } else if (raw.includes("402")) {
        msg = "Payment was not accepted. The transaction may have already been used."
      } else if (raw.includes("Failed to fetch") || raw.includes("NetworkError")) {
        msg = "Could not reach the server. Make sure it's running."
      } else {
        msg = "Something went wrong. Check the console for details."
        console.error("x402 SDK error:", raw)
      }
      setError(msg)
      addLog(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleCallApi = usePetraWallet ? handleCallApiPetra : handleCallApiPrivateKey

  const isConnected = usePetraWallet ? !!petraAddress : !!client
  const displayAddress = usePetraWallet ? petraAddress : walletAddress

  const [selectedWalletAddr, setSelectedWalletAddr] = useState<string | null>(null)

  const fillTestWallet = (wallet: typeof TEST_WALLETS[0]) => {
    setSelectedWalletAddr(wallet.address)
    setError(null)
  }

  return (
    <div className="container mx-auto py-8 px-4 max-w-4xl">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold mb-1 bg-gradient-to-r from-teal-500 to-emerald-500 bg-clip-text text-transparent">
          x402 Payment Demo
        </h1>
        <p className="text-muted-foreground text-sm">
          Pick a test wallet, call an API, watch the payment flow happen in real time.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Configuration Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wallet className="h-5 w-5" />
              1. Connect Wallet
            </CardTitle>
            <CardDescription>
              {usePetraWallet
                ? "Connect your Petra browser wallet"
                : "Use a pre-funded test wallet to get started instantly"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Wallet Mode Toggle */}
            <div className="flex items-center justify-between rounded-lg border p-3">
              <div className="space-y-0.5">
                <Label className="flex items-center gap-2">
                  <Wallet className="h-4 w-4 text-purple-500" />
                  Use Petra Wallet
                </Label>
                <p className="text-xs text-muted-foreground">
                  Browser extension (AIP-62)
                </p>
              </div>
              <Switch
                checked={usePetraWallet}
                onCheckedChange={(checked) => {
                  setUsePetraWallet(checked)
                  setClient(null)
                  setWalletAddress(null)
                  setBalance(null)
                  setLogs([])
                  setError(null)
                  setMilestone("idle")
                  if (!checked && walletConnected) {
                    walletDisconnect()
                  }
                }}
                disabled={isConnected}
              />
            </div>

            {/* Gasless Toggle — only for private key mode */}
            {!usePetraWallet && (
              <div className="flex items-center justify-between rounded-lg border p-3">
                <div className="space-y-0.5">
                  <Label className="flex items-center gap-2">
                    <Zap className="h-4 w-4 text-yellow-500" />
                    Gasless Mode
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Requires compatible facilitator
                  </p>
                </div>
                <Switch
                  checked={gasless}
                  onCheckedChange={setGasless}
                  disabled={!!client}
                />
              </div>
            )}

            {/* Test Wallets — promoted, shown first */}
            {!client && !usePetraWallet && (
              <div className="space-y-2">
                <Label className="text-xs font-medium">Test wallets — select one, then paste its private key below:</Label>
                <div className="flex flex-col gap-2">
                  {TEST_WALLETS.map((wallet) => (
                    <Button
                      key={wallet.name}
                      variant={selectedWalletAddr === wallet.address ? "default" : "outline"}
                      size="sm"
                      className="justify-start text-xs"
                      onClick={() => fillTestWallet(wallet)}
                    >
                      {wallet.name}
                      <span className="ml-auto font-mono text-[10px] opacity-60">
                        {wallet.address.slice(0, 8)}...{wallet.address.slice(-4)}
                      </span>
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Private Key Input */}
            {!usePetraWallet && (
              <div className="space-y-2">
                <Label htmlFor="private-key" className="text-xs text-muted-foreground">Or paste a private key:</Label>
                <div className="flex gap-2">
                  <Input
                    id="private-key"
                    type="password"
                    placeholder="0x..."
                    value={privateKey}
                    onChange={(e) => setPrivateKey(e.target.value)}
                    disabled={!!client}
                    className="font-mono text-xs"
                  />
                  {!client ? (
                    <Button onClick={handleConnectPrivateKey} disabled={connecting}>
                      {connecting ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        "Connect"
                      )}
                    </Button>
                  ) : (
                    <Button
                      variant="outline"
                      onClick={() => {
                        setClient(null)
                        setWalletAddress(null)
                        setBalance(null)
                        setLogs([])
                        setMilestone("idle")
                      }}
                    >
                      Disconnect
                    </Button>
                  )}
                </div>
              </div>
            )}

            {/* Petra Connect Button */}
            {usePetraWallet && !petraAddress && (
              <Button
                className="w-full"
                onClick={handleConnectPetra}
                disabled={connecting}
              >
                {connecting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <Wallet className="mr-2 h-4 w-4" />
                    Connect Petra Wallet
                  </>
                )}
              </Button>
            )}

            {/* Petra Disconnect */}
            {usePetraWallet && petraAddress && (
              <Button
                variant="outline"
                className="w-full"
                onClick={() => {
                  walletDisconnect()
                  setWalletAddress(null)
                  setLogs([])
                  setMilestone("idle")
                }}
              >
                Disconnect Petra
              </Button>
            )}

            {/* Wallet Info */}
            {displayAddress && (
              <div className="rounded-lg bg-muted p-3 space-y-1">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                  <span className="text-xs font-mono truncate">{displayAddress}</span>
                </div>
                {balance && (
                  <p className="text-xs text-muted-foreground pl-5">
                    {(Number(balance) / 100000000).toFixed(4)} APT ({balance} octas)
                  </p>
                )}
                {usePetraWallet && (
                  <p className="text-xs text-muted-foreground pl-5">
                    Petra will ask you to approve each payment.
                  </p>
                )}
              </div>
            )}

            {error && (
              <div className="flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/5 p-3 text-sm text-red-400">
                <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* API Call Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              2. Call a Paid API
            </CardTitle>
            <CardDescription>
              Select an endpoint — payment happens automatically
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Endpoint Selection */}
            <div className="space-y-2">
              <Label>Endpoint</Label>
              <Select value={selectedEndpoint} onValueChange={setSelectedEndpoint}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ENDPOINTS.map((ep) => (
                    <SelectItem key={ep.value} value={ep.value}>
                      <div className="flex justify-between w-full">
                        <span>{ep.label}</span>
                        <Badge variant="secondary" className="ml-2">{ep.price}</Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Symbol Input */}
            <div className="space-y-2">
              <Label htmlFor="symbol">Stock Symbol</Label>
              <Input
                id="symbol"
                placeholder="AAPL"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              />
            </div>

            {/* Call Button */}
            <Button
              className="w-full"
              onClick={handleCallApi}
              disabled={!isConnected || loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <ArrowRight className="mr-2 h-4 w-4" />
                  Call Protected API
                </>
              )}
            </Button>

            {!isConnected && (
              <p className="text-xs text-muted-foreground text-center">
                Connect a wallet first to enable API calls
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Live x402 Flow ── */}
      {milestone !== "idle" && (
        <Card className="mt-6">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Zap className="h-4 w-4 text-teal-500" />
              Live x402 Flow
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Milestone bar */}
            <MilestoneBar current={milestone} />

            <div className="grid gap-4 md:grid-cols-2">
              {/* Logs */}
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5 block">Event Log</Label>
                <pre className="bg-muted p-3 rounded-lg text-xs overflow-auto max-h-52 leading-relaxed">
                  {logs.join("\n")}
                </pre>
              </div>

              {/* Result */}
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5 block">
                  {result ? "API Response" : "Waiting for data..."}
                </Label>
                {result ? (
                  <div>
                    <pre className="bg-muted p-3 rounded-lg text-xs overflow-auto max-h-52 leading-relaxed">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                    {txHash && (
                      <a
                        href={`https://explorer.aptoslabs.com/txn/${txHash}?network=testnet`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-teal-500 hover:underline mt-2 inline-block"
                      >
                        View transaction on Aptos Explorer
                      </a>
                    )}
                  </div>
                ) : (
                  <div className="bg-muted p-3 rounded-lg text-xs text-muted-foreground h-20 flex items-center justify-center">
                    {loading ? "Payment in progress..." : "Run a call to see the response here."}
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── On-Chain Agent Identity ── */}
      <Card className="mt-6 border-purple-500/20">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm flex items-center gap-2">
                <Activity className="h-4 w-4 text-purple-500" />
                On-Chain Agent Identity
                {reputation?.registered && (
                  <Badge variant="secondary" className="ml-2 bg-green-500/10 text-green-500 border-green-500/20">
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                    Verified
                  </Badge>
                )}
              </CardTitle>
              <CardDescription className="text-xs mt-1">
                Autonomous agent in the Aptos agent economy
              </CardDescription>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={fetchReputation}
              disabled={reputationLoading}
            >
              <RefreshCw className={`h-3 w-3 ${reputationLoading ? "animate-spin" : ""}`} />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {reputation ? (
            <>
              {/* Agent Identity Section */}
              {reputation.registered && (
                <div className="rounded-lg bg-gradient-to-r from-purple-500/5 to-teal-500/5 border border-purple-500/10 p-4">
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <div className="text-xs text-muted-foreground">Agent Address</div>
                      <code className="text-xs font-mono bg-muted px-2 py-1 rounded">
                        {reputation.address?.slice(0, 10)}...{reputation.address?.slice(-8)}
                      </code>
                    </div>
                    <div className="text-right space-y-1">
                      <div className="text-xs text-muted-foreground">Capabilities</div>
                      <div className="text-lg font-bold text-purple-500">{reputation.capabilities}</div>
                    </div>
                  </div>
                  {reputation.endpointUrl && (
                    <div className="mt-3 pt-3 border-t border-purple-500/10">
                      <div className="text-xs text-muted-foreground mb-1">Service Endpoint</div>
                      <div className="text-xs font-mono text-teal-500">{reputation.endpointUrl}</div>
                    </div>
                  )}
                </div>
              )}

              {/* Stats Grid */}
              <div className="grid grid-cols-3 gap-4">
                <div className="rounded-lg border p-3 text-center">
                  <div className="text-2xl font-bold text-teal-500">{reputation.totalRequests}</div>
                  <div className="text-xs text-muted-foreground mt-1">Total Requests</div>
                </div>
                <div className="rounded-lg border p-3 text-center">
                  <div className="flex items-center justify-center gap-1">
                    <TrendingUp className="h-4 w-4 text-green-500" />
                    <span className="text-2xl font-bold text-green-500">{reputation.completionRate}%</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Success Rate</div>
                </div>
                <div className="rounded-lg border p-3 text-center">
                  <div className="text-2xl font-bold text-amber-500">{reputation.totalEarnedAPT}</div>
                  <div className="text-xs text-muted-foreground mt-1">APT Earned</div>
                </div>
              </div>

              {/* Explorer Link */}
              {reputation.explorerUrl && (
                <div className="text-center pt-2">
                  <a
                    href={reputation.explorerUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-purple-500 hover:underline inline-flex items-center gap-1"
                  >
                    View agent on Aptos Explorer →
                  </a>
                </div>
              )}
            </>
          ) : (
            <div className="text-xs text-muted-foreground text-center py-4">
              {reputationLoading ? "Loading on-chain data..." : "Could not load agent data."}
            </div>
          )}
        </CardContent>
      </Card>

    </div>
  )
}
