"use client"

/**
 * Usage Analytics Dashboard
 * 
 * Displays real-time and historical usage statistics for the x402 payment system:
 * - Total requests and revenue
 * - Endpoint usage breakdown
 * - Payment verification methods
 * - Recent transactions
 */

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  BarChart3, 
  TrendingUp, 
  CreditCard, 
  Activity,
  Zap,
  Globe,
  Clock
} from "lucide-react"

// In-memory analytics (would be from database in production)
interface UsageStats {
  totalRequests: number
  totalRevenue: number
  gaslessRequests: number
  directRequests: number
  endpointBreakdown: Record<string, number>
  recentTransactions: Transaction[]
}

interface Transaction {
  txHash: string
  endpoint: string
  amount: number
  timestamp: number
  mode: "gasless" | "direct"
  status: "success" | "pending" | "failed"
}

export default function UsageDashboardPage() {
  const [stats, setStats] = useState<UsageStats>({
    totalRequests: 0,
    totalRevenue: 0,
    gaslessRequests: 0,
    directRequests: 0,
    endpointBreakdown: {},
    recentTransactions: [],
  })

  // Simulate loading stats
  useEffect(() => {
    // In production, this would fetch from your database/API
    setStats({
      totalRequests: 1247,
      totalRevenue: 125600, // in octas
      gaslessRequests: 892,
      directRequests: 355,
      endpointBreakdown: {
        "/api/agent/momentum": 342,
        "/api/agent/stock-score": 298,
        "/api/agent/support-resistance": 256,
        "/api/analyze": 187,
        "/api/fundamentals": 164,
      },
      recentTransactions: [
        {
          txHash: "0xabc...123",
          endpoint: "/api/agent/momentum",
          amount: 100,
          timestamp: Date.now() - 1000 * 60 * 5,
          mode: "gasless",
          status: "success",
        },
        {
          txHash: "0xdef...456",
          endpoint: "/api/agent/stock-score",
          amount: 200,
          timestamp: Date.now() - 1000 * 60 * 12,
          mode: "direct",
          status: "success",
        },
        {
          txHash: "0xghi...789",
          endpoint: "/api/analyze",
          amount: 500,
          timestamp: Date.now() - 1000 * 60 * 23,
          mode: "gasless",
          status: "success",
        },
      ],
    })
  }, [])

  const formatOctas = (octas: number) => {
    return `${(octas / 100000000).toFixed(6)} APT`
  }

  const formatTime = (timestamp: number) => {
    const diff = Date.now() - timestamp
    const minutes = Math.floor(diff / 1000 / 60)
    if (minutes < 1) return "Just now"
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    return `${Math.floor(hours / 24)}d ago`
  }

  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-teal-500 to-emerald-500 bg-clip-text text-transparent">
          Usage Analytics Dashboard
        </h1>
        <p className="text-muted-foreground">
          Real-time insights into your x402 payment system
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-4 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalRequests.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              Across all endpoints
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Revenue</CardTitle>
            <CreditCard className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatOctas(stats.totalRevenue)}</div>
            <p className="text-xs text-muted-foreground">
              From micropayments
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Gasless Transactions</CardTitle>
            <Zap className="h-4 w-4 text-yellow-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.gaslessRequests.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              {((stats.gaslessRequests / stats.totalRequests) * 100).toFixed(1)}% of total
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg. Response Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">245ms</div>
            <p className="text-xs text-muted-foreground">
              Including verification
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="endpoints">Endpoints</TabsTrigger>
          <TabsTrigger value="transactions">Transactions</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Verification Methods */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Verification Methods
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">Gasless (Facilitator)</span>
                      <span className="text-sm text-muted-foreground">
                        {stats.gaslessRequests} requests
                      </span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-yellow-500 rounded-full"
                        style={{ 
                          width: `${(stats.gaslessRequests / stats.totalRequests) * 100}%` 
                        }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">Direct (On-Chain)</span>
                      <span className="text-sm text-muted-foreground">
                        {stats.directRequests} requests
                      </span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-500 rounded-full"
                        style={{ 
                          width: `${(stats.directRequests / stats.totalRequests) * 100}%` 
                        }}
                      />
                    </div>
                  </div>
                </div>

                <div className="mt-6 p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-foreground">Gasless Savings:</strong> Users saved 
                    approximately {formatOctas(stats.gaslessRequests * 500)} in gas fees 
                    through facilitator-sponsored transactions.
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Network Status */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="h-5 w-5" />
                  Network Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Network</span>
                    <Badge variant="secondary">Aptos Testnet</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Recipient Address</span>
                    <code className="text-xs bg-muted px-2 py-1 rounded">
                      0xaaef...42a0
                    </code>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Facilitator</span>
                    <Badge variant="outline" className="text-green-500">
                      Active
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Verification Mode</span>
                    <Badge variant="secondary">Hybrid</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Protected Endpoints</span>
                    <span className="font-medium">9</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="endpoints">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Endpoint Usage
              </CardTitle>
              <CardDescription>
                Request distribution across protected endpoints
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(stats.endpointBreakdown)
                  .sort(([,a], [,b]) => b - a)
                  .map(([endpoint, count]) => (
                    <div key={endpoint} className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">{endpoint}</span>
                          <span className="text-sm text-muted-foreground">
                            {count} requests ({((count / stats.totalRequests) * 100).toFixed(1)}%)
                          </span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-teal-500 rounded-full"
                            style={{ 
                              width: `${(count / stats.totalRequests) * 100}%` 
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="transactions">
          <Card>
            <CardHeader>
              <CardTitle>Recent Transactions</CardTitle>
              <CardDescription>
                Latest payment transactions processed
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {stats.recentTransactions.map((tx, i) => (
                  <div 
                    key={i}
                    className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${
                        tx.status === "success" ? "bg-green-500" : 
                        tx.status === "pending" ? "bg-yellow-500" : "bg-red-500"
                      }`} />
                      <div>
                        <p className="text-sm font-medium">{tx.endpoint}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatTime(tx.timestamp)}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">{tx.amount} octas</p>
                      <div className="flex items-center gap-1 justify-end">
                        {tx.mode === "gasless" ? (
                          <Zap className="h-3 w-3 text-yellow-500" />
                        ) : null}
                        <span className="text-xs text-muted-foreground capitalize">
                          {tx.mode}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Footer */}
      <div className="mt-8 text-center text-sm text-muted-foreground">
        <p>
          Powered by x402 on Aptos • Gasless transactions via facilitator
        </p>
      </div>
    </div>
  )
}
