/**
 * Minimal Bazaar Indexer Service
 * 
 * Responsibilities:
 * 1. Listen to on-chain events from Agent Registry
 * 2. Build/update search index
 * 3. Provide query API
 * 
 * Why separate service?
 * - On-chain queries are expensive/slow
 * - Bazaar needs fast search/filter
 * - This is a cache, source of truth is chain
 */

const express = require('express')
const { Aptos, AptosConfig, Network } = require('@aptos-labs/ts-sdk')

const app = express()
app.use(express.json())

// ==================== CONFIG ====================
const CONFIG = {
  NETWORK: Network.TESTNET,
  REGISTRY_ADDRESS: process.env.REGISTRY_ADDRESS || '0x1234...', // Replace with deployed address
  SYNC_INTERVAL_MS: 30000, // 30 seconds
  PORT: process.env.PORT || 4000
}

// ==================== IN-MEMORY INDEX ====================
// In production, use Redis or a proper database
const index = {
  agents: new Map(),              // address -> agent data
  byCapability: new Map(),        // capability -> [agents]
  byPriceRange: [],               // Sorted array for price queries
  lastSync: 0
}

// ==================== APTOS CLIENT ====================
const aptos = new Aptos(new AptosConfig({ 
  network: CONFIG.NETWORK 
}))

// ==================== SYNC LOGIC ====================

async function syncFromChain() {
  console.log('[Indexer] Syncing from chain...')
  
  try {
    // Get all registered agents via events
    // In production, you'd use a proper indexer or GraphQL
    const events = await aptos.getAccountEvents({
      accountAddress: CONFIG.REGISTRY_ADDRESS,
      eventType: `${CONFIG.REGISTRY_ADDRESS}::minimal_registry::AgentRegistered`
    })
    
    for (const event of events) {
      const agentAddress = event.data.agent
      
      // Fetch full agent data from chain
      const agentData = await aptos.view({
        payload: {
          function: `${CONFIG.REGISTRY_ADDRESS}::minimal_registry::get_agent`,
          functionArguments: [agentAddress]
        }
      })
      
      if (agentData && agentData[0]) {
        const agent = agentData[0]
        
        // Build indexed agent object
        const indexedAgent = {
          address: agentAddress,
          endpoint_url: agent.endpoint_url,
          capabilities: agent.capabilities.map(c => ({
            name: c.name,
            price: parseInt(c.price),
            description: c.description
          })),
          reputation: {
            total_requests: parseInt(agent.reputation.total_requests),
            successful_requests: parseInt(agent.reputation.successful_requests),
            failed_requests: parseInt(agent.reputation.failed_requests),
            total_earned: parseInt(agent.reputation.total_earned),
            total_ratings: parseInt(agent.reputation.total_ratings),
            rating_sum: parseInt(agent.reputation.rating_sum)
          },
          updated_at: Date.now()
        }
        
        // Calculate derived stats
        indexedAgent.completion_rate = indexedAgent.reputation.total_requests > 0
          ? indexedAgent.reputation.successful_requests / indexedAgent.reputation.total_requests
          : 0
        
        indexedAgent.average_rating = indexedAgent.reputation.total_ratings > 0
          ? (indexedAgent.reputation.rating_sum / indexedAgent.reputation.total_ratings) / 100
          : 0
        
        // Index by address
        index.agents.set(agentAddress, indexedAgent)
        
        // Index by capability
        for (const cap of indexedAgent.capabilities) {
          if (!index.byCapability.has(cap.name)) {
            index.byCapability.set(cap.name, [])
          }
          
          // Remove old entry if exists
          const capList = index.byCapability.get(cap.name)
          const existingIndex = capList.findIndex(a => a.address === agentAddress)
          if (existingIndex >= 0) {
            capList.splice(existingIndex, 1)
          }
          
          // Add new entry
          capList.push({
            address: agentAddress,
            price: cap.price,
            completion_rate: indexedAgent.completion_rate,
            average_rating: indexedAgent.average_rating
          })
        }
      }
    }
    
    // Build price range index
    index.byPriceRange = Array.from(index.agents.values())
      .flatMap(a => a.capabilities.map(c => ({
        address: a.address,
        capability: c.name,
        price: c.price,
        reputation: a.reputation
      })))
      .sort((a, b) => a.price - b.price)
    
    index.lastSync = Date.now()
    console.log(`[Indexer] Sync complete: ${index.agents.size} agents indexed`)
    
  } catch (error) {
    console.error('[Indexer] Sync failed:', error.message)
  }
}

// ==================== API ROUTES ====================

/**
 * Search agents by criteria
 * GET /search?capability=momentum&max_price=200&min_reputation=0.8
 */
app.get('/search', (req, res) => {
  const { 
    capability, 
    max_price, 
    min_reputation = 0,
    sort_by = 'reputation',  // 'reputation', 'price', 'recent'
    limit = 20
  } = req.query
  
  let results = []
  
  if (capability) {
    // Search by capability
    const capAgents = index.byCapability.get(capability) || []
    results = capAgents.map(ca => ({
      ...index.agents.get(ca.address),
      matching_price: ca.price
    }))
  } else {
    // Return all agents
    results = Array.from(index.agents.values())
  }
  
  // Filter by price
  if (max_price) {
    results = results.filter(a => 
      a.capabilities.some(c => c.price <= parseInt(max_price))
    )
  }
  
  // Filter by reputation (completion rate)
  results = results.filter(a => a.completion_rate >= parseFloat(min_reputation))
  
  // Sort
  results.sort((a, b) => {
    if (sort_by === 'price') {
      const minA = Math.min(...a.capabilities.map(c => c.price))
      const minB = Math.min(...b.capabilities.map(c => c.price))
      return minA - minB
    }
    if (sort_by === 'recent') {
      return b.updated_at - a.updated_at
    }
    // Default: reputation
    if (a.completion_rate !== b.completion_rate) {
      return b.completion_rate - a.completion_rate
    }
    return b.average_rating - a.average_rating
  })
  
  // Paginate
  const page = parseInt(req.query.page) || 1
  const offset = (page - 1) * limit
  const paginated = results.slice(offset, offset + parseInt(limit))
  
  res.json({
    agents: paginated,
    total: results.length,
    page,
    pages: Math.ceil(results.length / limit),
    last_sync: index.lastSync
  })
})

/**
 * Get single agent details
 * GET /agent/:address
 */
app.get('/agent/:address', (req, res) => {
  const agent = index.agents.get(req.params.address)
  
  if (!agent) {
    return res.status(404).json({ error: 'Agent not found' })
  }
  
  res.json(agent)
})

/**
 * List all capabilities
 * GET /capabilities
 */
app.get('/capabilities', (req, res) => {
  const capabilities = {}
  
  for (const [capName, agents] of index.byCapability) {
    const prices = agents.map(a => a.price)
    capabilities[capName] = {
      agent_count: agents.length,
      min_price: Math.min(...prices),
      max_price: Math.max(...prices),
      avg_price: prices.reduce((a, b) => a + b, 0) / prices.length
    }
  }
  
  res.json({ capabilities })
})

/**
 * Health check
 * GET /health
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    agents_indexed: index.agents.size,
    capabilities: index.byCapability.size,
    last_sync: index.lastSync
  })
})

/**
 * Force re-sync
 * POST /sync
 */
app.post('/sync', async (req, res) => {
  await syncFromChain()
  res.json({ 
    status: 'synced',
    agents: index.agents.size,
    last_sync: index.lastSync
  })
})

// ==================== STARTUP ====================

async function main() {
  console.log('[Indexer] Starting Bazaar Indexer...')
  console.log(`[Indexer] Registry: ${CONFIG.REGISTRY_ADDRESS}`)
  console.log(`[Indexer] Network: ${CONFIG.NETWORK}`)
  
  // Initial sync
  await syncFromChain()
  
  // Periodic sync
  setInterval(syncFromChain, CONFIG.SYNC_INTERVAL_MS)
  
  // Start server
  app.listen(CONFIG.PORT, () => {
    console.log(`[Indexer] API listening on http://localhost:${CONFIG.PORT}`)
    console.log(`[Indexer] Endpoints:`)
    console.log(`  GET  /search?capability=X&max_price=Y`)
    console.log(`  GET  /agent/:address`)
    console.log(`  GET  /capabilities`)
    console.log(`  GET  /health`)
    console.log(`  POST /sync`)
  })
}

main().catch(console.error)
