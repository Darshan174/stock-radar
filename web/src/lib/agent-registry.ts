/**
 * On-Chain Agent Registry Client
 * 
 * This module interacts with the Move-based AgentRegistry contract
 * to register agents, record usage, and retrieve reputation data on-chain.
 * 
 * ## Overview
 * 
 * The AgentRegistry contract allows:
 * - Registering agent identity on-chain
 * - Recording API usage and revenue
 * - Tracking reputation (ratings)
 * - Discovering agents by address
 * 
 * ## Contract Address (Testnet)
 * Deployed at: 0x... (to be filled after deployment)
 */

import { Aptos, AptosConfig, Network, Account } from "@aptos-labs/ts-sdk"

// Contract address (update after deployment)
const CONTRACT_ADDRESS = process.env.AGENT_REGISTRY_ADDRESS || "0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240"

// Module name (minimal_registry is the deployed module)
const MODULE_NAME = "minimal_registry"

export interface AgentCapability {
  name: string
  endpoint: string
  price: number
  description: string
}

export interface AgentInfo {
  owner: string
  name: string
  description: string
  endpointUrl: string
  capabilities: AgentCapability[]
  totalRequests: number
  successfulRequests: number
  failedRequests: number
  totalRevenue: number
  rating: number
  createdAt: number
  updatedAt: number
  isActive: boolean
  tags: string[]
}

export interface RegisterAgentParams {
  name: string
  description: string
  endpointUrl: string
  tags: string[]
}

export interface AddCapabilityParams {
  name: string
  endpoint: string
  price: number
  description: string
}

/**
 * Agent Registry Client
 * 
 * @example
 * ```typescript
 * const registry = new AgentRegistryClient("testnet")
 * 
 * // Register agent
 * await registry.registerAgent(account, {
 *   name: "Stock-Radar",
 *   description: "Financial Intelligence Agent",
 *   endpointUrl: "https://api.stockradar.io",
 *   tags: ["finance", "stocks", "ai"]
 * })
 * ```
 */
export class AgentRegistryClient {
  private aptos: Aptos

  constructor(network: Network = Network.TESTNET) {
    const config = new AptosConfig({ network })
    this.aptos = new Aptos(config)
  }

  /**
   * Register a new agent on-chain
   */
  async registerAgent(
    account: Account,
    params: RegisterAgentParams
  ): Promise<string> {
    const transaction = await this.aptos.transaction.build.simple({
      sender: account.accountAddress,
      data: {
        function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::register_agent`,
        functionArguments: [
          params.name,
          params.description,
          params.endpointUrl,
          params.tags,
        ],
      },
    })

    const pendingTxn = await this.aptos.signAndSubmitTransaction({
      signer: account,
      transaction,
    })

    await this.aptos.waitForTransaction({ transactionHash: pendingTxn.hash })
    return pendingTxn.hash
  }

  /**
   * Add a capability to the agent
   */
  async addCapability(
    account: Account,
    params: AddCapabilityParams
  ): Promise<string> {
    const transaction = await this.aptos.transaction.build.simple({
      sender: account.accountAddress,
      data: {
        function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::add_capability`,
        functionArguments: [
          params.name,
          params.endpoint,
          params.price,
          params.description,
        ],
      },
    })

    const pendingTxn = await this.aptos.signAndSubmitTransaction({
      signer: account,
      transaction,
    })

    await this.aptos.waitForTransaction({ transactionHash: pendingTxn.hash })
    return pendingTxn.hash
  }

  /**
   * Record an API request on-chain
   */
  async recordRequest(
    account: Account,
    agentOwner: string,
    capabilityName: string,
    amount: number,
    success: boolean
  ): Promise<string> {
    const transaction = await this.aptos.transaction.build.simple({
      sender: account.accountAddress,
      data: {
        function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::record_request`,
        functionArguments: [
          agentOwner,
          capabilityName,
          amount,
          success,
        ],
      },
    })

    const pendingTxn = await this.aptos.signAndSubmitTransaction({
      signer: account,
      transaction,
    })

    await this.aptos.waitForTransaction({ transactionHash: pendingTxn.hash })
    return pendingTxn.hash
  }

  /**
   * Submit a rating for an agent
   */
  async submitRating(
    account: Account,
    agentOwner: string,
    rating: number
  ): Promise<string> {
    const transaction = await this.aptos.transaction.build.simple({
      sender: account.accountAddress,
      data: {
        function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::submit_rating`,
        functionArguments: [agentOwner, rating],
      },
    })

    const pendingTxn = await this.aptos.signAndSubmitTransaction({
      signer: account,
      transaction,
    })

    await this.aptos.waitForTransaction({ transactionHash: pendingTxn.hash })
    return pendingTxn.hash
  }

  /**
   * Get agent info from on-chain
   */
  async getAgentInfo(owner: string): Promise<AgentInfo | null> {
    try {
      const result = await this.aptos.view({
        payload: {
          function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::get_agent`,
          functionArguments: [owner],
        },
      })

      if (!result || result.length === 0) return null

      const data = result[0] as any
      return this.parseAgentInfo(data)
    } catch (error) {
      console.error("Failed to get agent info:", error)
      return null
    }
  }

  /**
   * Get agent rating
   */
  async getAgentRating(owner: string): Promise<number> {
    try {
      const result = await this.aptos.view({
        payload: {
          function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::get_agent_rating`,
          functionArguments: [owner],
        },
      })

      return result[0] as number
    } catch (error) {
      return 0
    }
  }

  /**
   * Check if agent is registered
   */
  async isAgentRegistered(owner: string): Promise<boolean> {
    try {
      const result = await this.aptos.view({
        payload: {
          function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::is_agent_registered`,
          functionArguments: [owner],
        },
      })

      return result[0] as boolean
    } catch (error) {
      return false
    }
  }

  /**
   * Get total agent count
   */
  async getAgentCount(): Promise<number> {
    try {
      const result = await this.aptos.view({
        payload: {
          function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::get_agent_count`,
          functionArguments: [],
        },
      })

      return result[0] as number
    } catch (error) {
      return 0
    }
  }

  /**
   * Parse agent info from contract response
   */
  private parseAgentInfo(data: any): AgentInfo {
    const rep = data.reputation || {}
    return {
      owner: data.owner,
      name: data.name || "",
      description: data.description || "",
      endpointUrl: data.endpoint_url,
      capabilities: (data.capabilities || []).map((cap: any) => ({
        name: cap.name,
        endpoint: cap.endpoint || "",
        price: parseInt(cap.price),
        description: cap.description,
      })),
      totalRequests: parseInt(rep.total_requests) || 0,
      successfulRequests: parseInt(rep.successful_requests) || 0,
      failedRequests: parseInt(rep.failed_requests) || 0,
      totalRevenue: parseInt(rep.total_earned) || 0,
      rating: parseInt(rep.total_ratings) > 0 ? parseInt(rep.rating_sum) / parseInt(rep.total_ratings) : 0,
      createdAt: parseInt(rep.updated_at) || 0,
      updatedAt: parseInt(rep.updated_at) || 0,
      isActive: data.is_active ?? true,
      tags: data.tags || [],
    }
  }

  /**
   * Get on-chain reputation stats for an agent (view function, no signing required)
   */
  async getReputation(agentAddress: string): Promise<{
    totalRequests: number
    successfulRequests: number
    failedRequests: number
    totalEarned: number
    totalRatings: number
    ratingSUM: number
    updatedAt: number
  } | null> {
    try {
      const result = await this.aptos.view({
        payload: {
          function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::get_reputation`,
          functionArguments: [agentAddress],
        },
      })

      if (!result || result.length === 0) return null

      const data = result[0] as any
      return {
        totalRequests: parseInt(data.total_requests),
        successfulRequests: parseInt(data.successful_requests),
        failedRequests: parseInt(data.failed_requests),
        totalEarned: parseInt(data.total_earned),
        totalRatings: parseInt(data.total_ratings),
        ratingSUM: parseInt(data.rating_sum),
        updatedAt: parseInt(data.updated_at),
      }
    } catch (error) {
      console.error("Failed to get reputation:", error)
      return null
    }
  }

  /**
   * Update reputation on the minimal_registry contract
   * Matches: update_reputation(account, agent_address, success, earned, rating)
   */
  async updateReputation(
    account: Account,
    agentAddress: string,
    success: boolean,
    earned: number,
    rating: number = 0
  ): Promise<string> {
    const transaction = await this.aptos.transaction.build.simple({
      sender: account.accountAddress,
      data: {
        function: `${CONTRACT_ADDRESS}::${MODULE_NAME}::update_reputation`,
        functionArguments: [agentAddress, success, earned, rating],
      },
    })

    const pendingTxn = await this.aptos.signAndSubmitTransaction({
      signer: account,
      transaction,
    })

    await this.aptos.waitForTransaction({ transactionHash: pendingTxn.hash })
    return pendingTxn.hash
  }
}

// Singleton instance (testnet)
export const agentRegistry = new AgentRegistryClient(Network.TESTNET)

/**
 * Fire-and-forget reputation update after successful x402 payment.
 * Requires AGENT_REGISTRY_PRIVATE_KEY env var to sign transactions.
 */
export async function recordReputationUpdate(
  endpoint: string,
  amount: number,
  success: boolean
): Promise<void> {
  const rawKey = process.env.AGENT_REGISTRY_PRIVATE_KEY
  if (!rawKey) {
    console.warn("[reputation] AGENT_REGISTRY_PRIVATE_KEY not set, skipping")
    return
  }

  try {
    const { Ed25519PrivateKey } = await import("@aptos-labs/ts-sdk")
    // Strip 0x prefix if present — Ed25519PrivateKey expects raw hex
    const hex = rawKey.startsWith("0x") ? rawKey.slice(2) : rawKey
    const privateKey = new Ed25519PrivateKey(hex)
    const account = Account.fromPrivateKey({ privateKey })
    const agentAddress = account.accountAddress.toString()

    console.log(`[reputation] Updating reputation for ${agentAddress}, endpoint=${endpoint}, amount=${amount}, success=${success}`)
    const txHash = await agentRegistry.updateReputation(account, agentAddress, success, amount, 0)
    console.log(`[reputation] Recorded ${success ? "success" : "failure"} for ${endpoint}, earned=${amount}, tx=${txHash}`)
  } catch (err: any) {
    console.error(`[reputation] Failed to record on-chain:`, err?.message || err)
  }
}
