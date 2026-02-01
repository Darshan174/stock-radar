/**
 * XMTP Agent Client Example
 * 
 * This shows how an agent or client would use XMTP for:
 * 1. Discovering capabilities
 * 2. Negotiating prices
 * 3. Requesting tasks
 * 4. Building trust networks
 */

const { AgentMessenger } = require('../xmtp-agent')
const { Wallet } = require('ethers')

// ==================== AGENT EXAMPLE ====================

async function runAgent() {
  console.log('=== Starting Stock Analysis Agent ===\n')
  
  // Agent wallet (fixed address)
  const wallet = Wallet.createRandom()
  console.log('Agent address:', wallet.address)
  
  // Create messenger
  const messenger = await AgentMessenger.create(wallet, { env: 'dev' })
  
  // Handle capability discovery
  messenger.on('capability_discovery', async (msg) => {
    console.log(`\n📩 Capability inquiry from ${msg.from.slice(0, 10)}...`)
    console.log('   Requested:', msg.payload.requested_capabilities.join(', '))
    
    // Check if we can fulfill
    const ourCapabilities = ['momentum', 'fundamentals', 'sentiment']
    const canFulfill = msg.payload.requested_capabilities.every(
      cap => ourCapabilities.includes(cap)
    )
    
    if (canFulfill) {
      console.log('   ✅ We can fulfill! Sending offer...')
      
      await messenger.sendOffer(msg.from, {
        capabilities: [
          {
            name: 'momentum',
            price: 100,
            endpoint: 'https://api.stockagent.com/momentum',
            estimatedTime: 500
          },
          {
            name: 'fundamentals',
            price: 150,
            endpoint: 'https://api.stockagent.com/fundamentals',
            estimatedTime: 1000
          },
          {
            name: 'sentiment',
            price: 200,
            endpoint: 'https://api.stockagent.com/sentiment',
            estimatedTime: 2000
          }
        ],
        bulkDiscount: {
          minRequests: 10,
          discountPercent: 15
        }
      })
    } else {
      console.log('   ❌ Cannot fulfill these capabilities')
    }
  })
  
  // Handle task requests
  messenger.on('task_request', async (msg) => {
    console.log(`\n📩 Task request from ${msg.from.slice(0, 10)}...`)
    console.log('   Task:', msg.payload.task_id)
    console.log('   Capability:', msg.payload.capability)
    console.log('   Max price:', msg.payload.max_price, 'octas')
    
    // Accept if price is reasonable
    const minPrice = 80  // Our minimum
    if (msg.payload.max_price >= minPrice) {
      console.log('   ✅ Accepting task...')
      
      await messenger.acceptTask(msg.from, {
        taskId: msg.payload.task_id,
        price: msg.payload.max_price,
        paymentAddress: wallet.address,
        endpoint: `https://api.stockagent.com/${msg.payload.capability}`,
        estimatedCompletion: Date.now() + 5000
      })
      
      console.log('   📤 Task accepted, waiting for x402 payment...')
    } else {
      console.log('   ❌ Price too low, rejecting')
    }
  })
  
  // Start listening
  await messenger.startListening()
  console.log('\n🟢 Agent is listening for messages...')
}

// ==================== CLIENT EXAMPLE ====================

async function runClient(agentAddress) {
  console.log('\n=== Starting Client ===\n')
  
  // Client wallet
  const wallet = Wallet.createRandom()
  console.log('Client address:', wallet.address)
  
  // Create messenger
  const messenger = await AgentMessenger.create(wallet, { env: 'dev' })
  
  // Set up handlers for responses
  messenger.on('offer', (msg) => {
    console.log('\n📩 Received offer from agent:')
    msg.payload.capabilities.forEach(cap => {
      console.log(`   • ${cap.name}: ${cap.price} octas (${cap.estimated_ms}ms)`)
    })
    
    if (msg.payload.bulk_discount) {
      console.log(`   💰 Bulk discount: ${msg.payload.bulk_discount.discount_percent}% off for ${msg.payload.bulk_discount.min_requests}+ requests`)
    }
  })
  
  messenger.on('task_acceptance', (msg) => {
    console.log('\n📩 Task accepted!')
    console.log('   Agreed price:', msg.payload.agreed_price)
    console.log('   Payment address:', msg.payload.payment_address)
    console.log('   x402 endpoint:', msg.payload.endpoint)
    console.log('   ETA:', new Date(msg.payload.estimated_completion).toLocaleTimeString())
    
    // Now the client would:
    // 1. Make HTTP request to endpoint with x402 payment
    // 2. Get analysis result
    console.log('\n   💡 Next: Call', msg.payload.endpoint, 'with x402 payment')
  })
  
  // Start listening
  messenger.startListening()
  
  // Wait a moment then send inquiry
  setTimeout(async () => {
    console.log(`\n📤 Sending capability inquiry to ${agentAddress.slice(0, 10)}...`)
    await messenger.inquireCapabilities(agentAddress, {
      capabilities: ['momentum', 'fundamentals'],
      budget: { min: 50, max: 300 },
      deadline: Date.now() + 3600000
    })
    
    // Wait then send task request
    setTimeout(async () => {
      console.log('\n📤 Sending task request...')
      await messenger.requestTask(agentAddress, {
        id: `task-${Date.now()}`,
        capability: 'momentum',
        parameters: { symbol: 'AAPL' },
        maxPrice: 150,
        deadline: Date.now() + 60000
      })
    }, 3000)
    
  }, 2000)
}

// ==================== TRUST NETWORK EXAMPLE ====================

async function vouchExample() {
  console.log('\n=== Trust Network Demo ===\n')
  
  // Agent A vouches for Agent B
  const agentA = await AgentMessenger.create(Wallet.createRandom(), { env: 'dev' })
  const agentB = Wallet.createRandom().address
  
  console.log('Agent A:', agentA.client.address)
  console.log('Agent B:', agentB)
  
  // Agent B sets up handler
  agentA.on('reputation_vouch', (msg) => {
    console.log('\n📩 Received reputation vouch!')
    console.log('   From:', msg.payload.voucher.slice(0, 10) + '...')
    console.log('   Rating:', msg.payload.rating, '/ 5')
    console.log('   Completed tasks together:', msg.payload.completed_tasks)
    console.log('   Comment:', msg.payload.comment)
    
    // Agent B can use this vouch in their reputation
    console.log('   ✅ Vouch stored for reputation building')
  })
  
  await agentA.startListening()
  
  // Agent A sends vouch
  setTimeout(async () => {
    console.log('\n📤 Agent A vouching for Agent B...')
    await agentA.vouchForAgent(agentB, {
      rating: 5,
      completedTasks: 12,
      comment: 'Excellent momentum analysis, always on time'
    })
  }, 1000)
}

// ==================== MAIN ====================

async function main() {
  const [mode] = process.argv.slice(2)
  
  if (mode === 'agent') {
    await runAgent()
  } else if (mode === 'client') {
    if (process.argv.length < 4) {
      console.log('Usage: node example.js client <agent_address>')
      process.exit(1)
    }
    await runClient(process.argv[3])
  } else if (mode === 'vouch') {
    await vouchExample()
  } else {
    console.log('Usage:')
    console.log('  node example.js agent              # Run as agent')
    console.log('  node example.js client <address>   # Run as client')
    console.log('  node example.js vouch              # Run trust demo')
  }
}

main().catch(console.error)
