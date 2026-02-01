// Minimal Agent Registry for x402
// Addresses: Common mistakes, verification, idempotency

module agent_registry::minimal_registry {
    use std::string::{String};
    use std::signer;
    use std::vector;
    use aptos_std::table::{Self, Table};
    use aptos_framework::timestamp;
    use aptos_framework::event;

    // ============ ERRORS ============
    const E_AGENT_EXISTS: u64 = 1;
    const E_AGENT_NOT_FOUND: u64 = 2;
    const E_UNAUTHORIZED: u64 = 3;

    // ============ DATA STRUCTURES ============
    
    /// Capability with pricing - stored per-agent
    struct Capability has store, copy, drop {
        name: String,
        price: u64,          // In octas (1 APT = 100,000,000 octas)
        description: String,
    }

    /// Agent reputation counters - updated ONLY after verified payments
    struct Reputation has store, copy, drop {
        total_requests: u64,
        successful_requests: u64,
        failed_requests: u64,
        total_earned: u64,       // Total octas earned
        total_ratings: u64,
        rating_sum: u64,         // For average calculation
        updated_at: u64,
    }

    /// Main Agent data
    struct Agent has store, copy, drop {
        owner: address,
        endpoint_url: String,
        capabilities: vector<Capability>,
        reputation: Reputation,
    }

    /// Global registry
    struct Registry has key {
        agents: Table<address, Agent>,
        admin: address,
    }

    // ============ EVENTS ============
    
    #[event]
    struct AgentRegistered has drop, store {
        agent: address,
        endpoint_url: String,
        timestamp: u64,
    }

    #[event]
    struct ReputationUpdated has drop, store {
        agent: address,
        success: bool,
        earned: u64,
        new_total_requests: u64,
        timestamp: u64,
    }

    #[event]
    struct CapabilityAdded has drop, store {
        agent: address,
        name: String,
        price: u64,
        timestamp: u64,
    }

    // ============ INIT ============
    
    fun init_module(account: &signer) {
        move_to(account, Registry {
            agents: table::new(),
            admin: signer::address_of(account),
        });
    }

    // ============ AGENT REGISTRATION ============
    
    /// Register a new agent with capabilities
    public entry fun register_agent(
        account: &signer,
        endpoint_url: String,
        capability_names: vector<String>,
        capability_prices: vector<u64>,
        capability_descriptions: vector<String>,
    ) acquires Registry {
        let owner = signer::address_of(account);
        let registry = borrow_global_mut<Registry>(@agent_registry);
        
        // Check agent doesn't exist
        assert!(!table::contains(&registry.agents, owner), E_AGENT_EXISTS);
        
        // Build capabilities
        let capabilities = vector::empty<Capability>();
        let i = 0;
        let len = vector::length(&capability_names);
        while (i < len) {
            vector::push_back(&mut capabilities, Capability {
                name: *vector::borrow(&capability_names, i),
                price: *vector::borrow(&capability_prices, i),
                description: *vector::borrow(&capability_descriptions, i),
            });
            i = i + 1;
        };
        
        // Create agent
        let agent = Agent {
            owner,
            endpoint_url,
            capabilities,
            reputation: Reputation {
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                total_earned: 0,
                total_ratings: 0,
                rating_sum: 0,
                updated_at: timestamp::now_seconds(),
            },
        };
        
        table::add(&mut registry.agents, owner, agent);
        
        event::emit(AgentRegistered {
            agent: owner,
            endpoint_url,
            timestamp: timestamp::now_seconds(),
        });
    }

    // ============ REPUTATION (UPDATED AFTER VERIFIED PAYMENT) ============
    
    /// Update reputation - MUST be called AFTER payment verification
    /// Only admin or the agent owner can update reputation
    public entry fun update_reputation(
        account: &signer,
        agent_address: address,
        success: bool,
        earned: u64,           // Amount earned in this transaction (octas)
        rating: u64,           // 0-500 (scaled by 100, 0 = no rating)
    ) acquires Registry {
        let caller = signer::address_of(account);
        let registry = borrow_global_mut<Registry>(@agent_registry);
        assert!(table::contains(&registry.agents, agent_address), E_AGENT_NOT_FOUND);

        // Only admin or the agent owner can update reputation
        assert!(caller == registry.admin || caller == agent_address, E_UNAUTHORIZED);
        
        let agent = table::borrow_mut(&mut registry.agents, agent_address);
        let rep = &mut agent.reputation;
        
        // Update counters
        rep.total_requests = rep.total_requests + 1;
        rep.total_earned = rep.total_earned + earned;
        rep.updated_at = timestamp::now_seconds();
        
        if (success) {
            rep.successful_requests = rep.successful_requests + 1;
        } else {
            rep.failed_requests = rep.failed_requests + 1;
        };
        
        // Update rating if provided
        if (rating > 0 && rating <= 500) {
            rep.total_ratings = rep.total_ratings + 1;
            rep.rating_sum = rep.rating_sum + rating;
        };
        
        event::emit(ReputationUpdated {
            agent: agent_address,
            success,
            earned,
            new_total_requests: rep.total_requests,
            timestamp: rep.updated_at,
        });
    }

    // ============ CAPABILITY MANAGEMENT ============
    
    /// Add a new capability
    public entry fun add_capability(
        account: &signer,
        name: String,
        price: u64,
        description: String,
    ) acquires Registry {
        let owner = signer::address_of(account);
        let registry = borrow_global_mut<Registry>(@agent_registry);
        assert!(table::contains(&registry.agents, owner), E_AGENT_NOT_FOUND);
        
        let agent = table::borrow_mut(&mut registry.agents, owner);
        
        vector::push_back(&mut agent.capabilities, Capability {
            name,
            price,
            description,
        });
        
        event::emit(CapabilityAdded {
            agent: owner,
            name,
            price,
            timestamp: timestamp::now_seconds(),
        });
    }

    // ============ VIEW FUNCTIONS ============
    
    /// Get full agent data
    #[view]
    public fun get_agent(agent_address: address): Agent acquires Registry {
        let registry = borrow_global<Registry>(@agent_registry);
        assert!(table::contains(&registry.agents, agent_address), E_AGENT_NOT_FOUND);
        *table::borrow(&registry.agents, agent_address)
    }

    /// Check if agent exists
    #[view]
    public fun agent_exists(agent_address: address): bool acquires Registry {
        let registry = borrow_global<Registry>(@agent_registry);
        table::contains(&registry.agents, agent_address)
    }

    /// Get capability price
    #[view]
    public fun get_price(agent_address: address, capability_name: String): u64 acquires Registry {
        let registry = borrow_global<Registry>(@agent_registry);
        let agent = table::borrow(&registry.agents, agent_address);
        
        let i = 0;
        let len = vector::length(&agent.capabilities);
        while (i < len) {
            let cap = vector::borrow(&agent.capabilities, i);
            if (cap.name == capability_name) {
                return cap.price
            };
            i = i + 1;
        };
        
        0  // Not found
    }

    /// Get reputation stats
    #[view]
    public fun get_reputation(agent_address: address): Reputation acquires Registry {
        let registry = borrow_global<Registry>(@agent_registry);
        let agent = table::borrow(&registry.agents, agent_address);
        agent.reputation
    }

    /// Calculate average rating (0-500 scaled)
    #[view]
    public fun get_average_rating(agent_address: address): u64 acquires Registry {
        let registry = borrow_global<Registry>(@agent_registry);
        let agent = table::borrow(&registry.agents, agent_address);
        
        if (agent.reputation.total_ratings == 0) {
            return 0
        };
        
        agent.reputation.rating_sum / agent.reputation.total_ratings
    }

    /// Calculate completion rate (0-10000, scaled by 100)
    #[view]
    public fun get_completion_rate(agent_address: address): u64 acquires Registry {
        let registry = borrow_global<Registry>(@agent_registry);
        let agent = table::borrow(&registry.agents, agent_address);
        
        if (agent.reputation.total_requests == 0) {
            return 0
        };
        
        (agent.reputation.successful_requests * 10000) / agent.reputation.total_requests
    }
}
