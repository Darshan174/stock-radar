/**
 * XMTP Agent Messenger
 * 
 * Why XMTP for agents?
 * - Wallet-based identity (no separate login)
 * - E2E encrypted messaging
 * - Cross-app compatibility
 * - Perfect for machine-to-machine negotiation
 * 
 * Use Cases:
 * 1. Capability discovery ("What can you do?")
 * 2. Pricing negotiation ("I'll do it for 80 octas")
 * 3. Task coordination ("You handle sentiment, I'll do momentum")
 * 4. Reputation vouching ("This agent is reliable")
 */

const { Client } = require('@xmtp/xmtp-js')
const { Wallet } = require('ethers')

// ==================== MESSAGE PROTOCOL ====================

/**
 * Standard message envelope for agent communication
 */
const PROTOCOL_VERSION = 'agent-xmtp-v1'

function createMessage(type, payload) {
  return {
    protocol: PROTOCOL_VERSION,
    message_type: type,
    timestamp: Date.now(),
    payload
  }
}

/**
 * Message Types:
 * 
 * 1. capability_discovery - Find what an agent can do
 * 2. pricing_inquiry - Ask about pricing
 * 3. offer - Respond with pricing
 * 4. task_request - Request work
 * 5. task_response - Accept/reject work
 * 6. reputation_vouch - Vouch for another agent
 */

// ==================== AGENT MESSENGER CLASS ====================

class AgentMessenger {
  constructor(xmtpClient) {
    this.client = xmtpClient
    this.conversations = new Map()  // address -> Conversation
    this.messageHandlers = new Map()  // type -> handler
  }

  /**
   * Create new messenger instance
   * @param {Object} wallet - Ethers wallet (uses same address as agent)
   */
  static async create(wallet, options = {}) {
    const client = await Client.create(wallet, {
      env: options.env || 'dev'  // 'dev' for testnet, 'production' for mainnet
    })
    
    console.log('[XMTP] Client created for:', client.address)
    return new AgentMessenger(client)
  }

  // ==================== SENDING MESSAGES ====================

  /**
   * Send capability discovery request
   * "What capabilities do you have and how much do they cost?"
   */
  async inquireCapabilities(agentAddress, requirements) {
    const message = createMessage('capability_discovery', {
      requested_capabilities: requirements.capabilities || [],
      budget_range: requirements.budget || { min: 0, max: 1000 },
      deadline: requirements.deadline || Date.now() + 3600000
    })

    return this.sendMessage(agentAddress, message)
  }

  /**
   * Send pricing inquiry
   * "I need X, what's your price?"
   */
  async inquirePricing(agentAddress, inquiry) {
    const message = createMessage('pricing_inquiry', {
      capability: inquiry.capability,
      estimated_volume: inquiry.volume || 1,  // Expected number of requests
      urgency: inquiry.urgency || 'normal'     // 'low', 'normal', 'high'
    })

    return this.sendMessage(agentAddress, message)
  }

  /**
   * Send capability offer
   * "I can do X for Y price"
   */
  async sendOffer(clientAddress, offer) {
    const message = createMessage('offer', {
      agent: this.client.address,
      capabilities: offer.capabilities.map(c => ({
        name: c.name,
        price: c.price,
        currency: c.currency || 'APT',
        endpoint: c.endpoint,
        estimated_ms: c.estimatedTime || 1000
      })),
      bulk_discount: offer.bulkDiscount || null,
      valid_until: offer.validUntil || Date.now() + 86400000  // 24 hours
    })

    return this.sendMessage(clientAddress, message)
  }

  /**
   * Send task request
   * "Please do this task"
   */
  async requestTask(agentAddress, task) {
    const message = createMessage('task_request', {
      task_id: task.id || `task-${Date.now()}`,
      capability: task.capability,
      parameters: task.parameters,
      max_price: task.maxPrice,
      deadline: task.deadline,
      callback_url: task.callbackUrl  // Optional webhook for completion
    })

    return this.sendMessage(agentAddress, message)
  }

  /**
   * Send task acceptance
   * "I'll do the task for X price"
   */
  async acceptTask(clientAddress, acceptance) {
    const message = createMessage('task_acceptance', {
      task_id: acceptance.taskId,
      agreed_price: acceptance.price,
      payment_address: acceptance.paymentAddress,
      estimated_completion: acceptance.estimatedCompletion,
      endpoint: acceptance.endpoint  // x402 endpoint to call
    })

    return this.sendMessage(clientAddress, message)
  }

  /**
   * Send reputation vouch
   * "I vouch for this agent"
   */
  async vouchForAgent(vouchedAddress, vouch) {
    // Send to the vouched agent (they can use this to prove reputation)
    const message = createMessage('reputation_vouch', {
      voucher: this.client.address,
      vouched_agent: vouchedAddress,
      rating: vouch.rating,  // 1-5
      completed_tasks: vouch.completedTasks || 0,
      comment: vouch.comment || '',
      timestamp: Date.now()
    })

    return this.sendMessage(vouchedAddress, message)
  }

  // ==================== INTERNAL METHODS ====================

  /**
   * Send raw message (internal)
   */
  async sendMessage(toAddress, message) {
    // Get or create conversation
    let conversation = this.conversations.get(toAddress)
    
    if (!conversation) {
      conversation = await this.client.conversations.newConversation(toAddress)
      this.conversations.set(toAddress, conversation)
      console.log(`[XMTP] New conversation with: ${toAddress}`)
    }

    const content = JSON.stringify(message)
    const result = await conversation.send(content)
    
    console.log(`[XMTP] Sent ${message.message_type} to ${toAddress}`)
    return result
  }

  // ==================== RECEIVING MESSAGES ====================

  /**
   * Register handler for specific message type
   */
  on(messageType, handler) {
    this.messageHandlers.set(messageType, handler)
  }

  /**
   * Start listening for messages
   */
  async startListening() {
    console.log('[XMTP] Starting message listener...')
    
    const stream = await this.client.conversations.streamAllMessages()
    
    for await (const msg of stream) {
      // Skip own messages
      if (msg.senderAddress === this.client.address) continue
      
      try {
        const parsed = JSON.parse(msg.content)
        
        // Only handle agent protocol messages
        if (parsed.protocol !== PROTOCOL_VERSION) continue
        
        console.log(`[XMTP] Received ${parsed.message_type} from ${msg.senderAddress}`)
        
        // Route to handler
        const handler = this.messageHandlers.get(parsed.message_type)
        if (handler) {
          handler({
            from: msg.senderAddress,
            type: parsed.message_type,
            payload: parsed.payload,
            timestamp: parsed.timestamp,
            messageId: msg.id
          })
        } else {
          console.log(`[XMTP] No handler for: ${parsed.message_type}`)
        }
        
      } catch (e) {
        // Not JSON or invalid format, skip
      }
    }
  }

  /**
   * Stop listening
   */
  stopListening() {
    // XMTP streams can't be cleanly stopped in current version
    // This would require implementing a cancellation token
    console.log('[XMTP] Stopping listener (not fully supported in XMTP v11)')
  }

  // ==================== CONVENIENCE METHODS ====================

  /**
   * Get all conversations
   */
  async getConversations() {
    return this.client.conversations.list()
  }

  /**
   * Get message history with specific agent
   */
  async getHistory(agentAddress, limit = 50) {
    const conversation = this.conversations.get(agentAddress)
    if (!conversation) return []
    
    return conversation.messages({ limit })
  }

  /**
   * Check if can message an address
   */
  async canMessage(address) {
    return Client.canMessage(this.client.address, address)
  }
}

// ==================== EXAMPLE USAGE ====================

async function exampleUsage() {
  // Create wallet (in production, use your actual agent wallet)
  const wallet = Wallet.createRandom()
  
  // Create messenger
  const messenger = await AgentMessenger.create(wallet)
  
  // Set up handlers
  messenger.on('capability_discovery', (msg) => {
    console.log(`Client ${msg.from} wants:`, msg.payload.requested_capabilities)
    
    // Send offer back
    messenger.sendOffer(msg.from, {
      capabilities: [
        {
          name: 'momentum',
          price: 100,
          endpoint: 'http://localhost:3000/api/agent/momentum',
          estimatedTime: 500
        }
      ],
      bulkDiscount: {
        minRequests: 10,
        discountPercent: 15
      }
    })
  })
  
  messenger.on('task_request', (msg) => {
    console.log(`Task request:`, msg.payload)
    
    // Accept the task
    messenger.acceptTask(msg.from, {
      taskId: msg.payload.task_id,
      price: msg.payload.max_price,
      paymentAddress: wallet.address,
      endpoint: 'http://localhost:3000/api/agent/momentum',
      estimatedCompletion: Date.now() + 5000
    })
  })
  
  // Start listening
  messenger.startListening()
  
  // Example: Proactively inquire about capabilities
  // await messenger.inquireCapabilities('0xabc...', {
  //   capabilities: ['momentum'],
  //   budget: { min: 50, max: 200 }
  // })
}

// Run example if executed directly
if (require.main === module) {
  exampleUsage().catch(console.error)
}

module.exports = { AgentMessenger, createMessage, PROTOCOL_VERSION }
