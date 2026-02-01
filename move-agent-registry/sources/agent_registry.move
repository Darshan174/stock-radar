module agent_registry::agent_registry {
    use std::string::{String, utf8};
    use std::signer;
    use std::vector;
    use aptos_std::table::{Self, Table};
    use aptos_framework::timestamp;
    use aptos_framework::event;

    /// Error codes
    const E_NOT_AUTHORIZED: u64 = 1;
    const E_AGENT_ALREADY_EXISTS: u64 = 2;
    const E_AGENT_NOT_FOUND: u64 = 3;
    const E_INVALID_PRICE: u64 = 4;

    /// Agent capabilities
    struct AgentCapability has store, copy, drop {
        name: String,
        endpoint: String,
        price: u64,
        description: String,
    }

    /// Agent information
    struct AgentInfo has store, copy, drop {
        owner: address,
        name: String,
        description: String,
        endpoint_url: String,
        capabilities: vector<AgentCapability>,
        total_requests: u64,
        successful_requests: u64,
        failed_requests: u64,
        total_revenue: u64,
        rating_sum: u64,
        rating_count: u64,
        created_at: u64,
        updated_at: u64,
        is_active: bool,
        tags: vector<String>,
    }

    /// Global registry
    struct AgentRegistry has key {
        agents: Table<address, AgentInfo>,
        agent_count: u64,
        admin: address,
    }

    /// Events
    #[event]
    struct AgentRegistered has drop, store {
        owner: address,
        name: String,
        endpoint_url: String,
        timestamp: u64,
    }

    #[event]
    struct AgentUpdated has drop, store {
        owner: address,
        name: String,
        timestamp: u64,
    }

    #[event]
    struct CapabilityAdded has drop, store {
        owner: address,
        capability_name: String,
        price: u64,
        timestamp: u64,
    }

    #[event]
    struct RequestRecorded has drop, store {
        owner: address,
        capability_name: String,
        amount: u64,
        success: bool,
        timestamp: u64,
    }

    #[event]
    struct RatingSubmitted has drop, store {
        agent: address,
        rater: address,
        rating: u8,
        timestamp: u64,
    }

    /// Initialize the registry
    fun init_module(account: &signer) {
        let admin = signer::address_of(account);
        move_to(account, AgentRegistry {
            agents: table::new(),
            agent_count: 0,
            admin,
        });
    }

    /// Register a new agent
    public entry fun register_agent(
        account: &signer,
        name: String,
        description: String,
        endpoint_url: String,
        tags: vector<String>,
    ) acquires AgentRegistry {
        let owner = signer::address_of(account);
        let registry = borrow_global_mut<AgentRegistry>(@agent_registry);
        
        assert!(!table::contains(&registry.agents, owner), E_AGENT_ALREADY_EXISTS);

        let agent = AgentInfo {
            owner,
            name,
            description,
            endpoint_url,
            capabilities: vector::empty(),
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_revenue: 0,
            rating_sum: 0,
            rating_count: 0,
            created_at: timestamp::now_seconds(),
            updated_at: timestamp::now_seconds(),
            is_active: true,
            tags,
        };

        table::add(&mut registry.agents, owner, agent);
        registry.agent_count = registry.agent_count + 1;

        event::emit(AgentRegistered {
            owner,
            name,
            endpoint_url,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Add a capability to an agent
    public entry fun add_capability(
        account: &signer,
        capability_name: String,
        endpoint: String,
        price: u64,
        description: String,
    ) acquires AgentRegistry {
        let owner = signer::address_of(account);
        let registry = borrow_global_mut<AgentRegistry>(@agent_registry);
        
        assert!(table::contains(&registry.agents, owner), E_AGENT_NOT_FOUND);
        assert!(price > 0, E_INVALID_PRICE);

        let agent = table::borrow_mut(&mut registry.agents, owner);
        
        let capability = AgentCapability {
            name: capability_name,
            endpoint,
            price,
            description,
        };
        
        vector::push_back(&mut agent.capabilities, capability);
        agent.updated_at = timestamp::now_seconds();

        event::emit(CapabilityAdded {
            owner,
            capability_name,
            price,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Record a request (called by the agent's server)
    public entry fun record_request(
        account: &signer,
        agent_owner: address,
        capability_name: String,
        amount: u64,
        success: bool,
    ) acquires AgentRegistry {
        // In production, this should be authenticated
        // For now, any caller can record (simplified for hackathon)
        let registry = borrow_global_mut<AgentRegistry>(@agent_registry);
        
        assert!(table::contains(&registry.agents, agent_owner), E_AGENT_NOT_FOUND);

        let agent = table::borrow_mut(&mut registry.agents, agent_owner);
        
        agent.total_requests = agent.total_requests + 1;
        agent.total_revenue = agent.total_revenue + amount;
        
        if (success) {
            agent.successful_requests = agent.successful_requests + 1;
        } else {
            agent.failed_requests = agent.failed_requests + 1;
        };

        agent.updated_at = timestamp::now_seconds();

        event::emit(RequestRecorded {
            owner: agent_owner,
            capability_name,
            amount,
            success,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Submit a rating for an agent
    public entry fun submit_rating(
        account: &signer,
        agent_owner: address,
        rating: u8,
    ) acquires AgentRegistry {
        let rater = signer::address_of(account);
        let registry = borrow_global_mut<AgentRegistry>(@agent_registry);
        
        assert!(table::contains(&registry.agents, agent_owner), E_AGENT_NOT_FOUND);
        assert!(rating >= 1 && rating <= 5, E_INVALID_PRICE);

        let agent = table::borrow_mut(&mut registry.agents, agent_owner);
        
        agent.rating_sum = agent.rating_sum + (rating as u64);
        agent.rating_count = agent.rating_count + 1;

        event::emit(RatingSubmitted {
            agent: agent_owner,
            rater,
            rating,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Update agent info
    public entry fun update_agent(
        account: &signer,
        name: String,
        description: String,
        endpoint_url: String,
        is_active: bool,
    ) acquires AgentRegistry {
        let owner = signer::address_of(account);
        let registry = borrow_global_mut<AgentRegistry>(@agent_registry);
        
        assert!(table::contains(&registry.agents, owner), E_AGENT_NOT_FOUND);

        let agent = table::borrow_mut(&mut registry.agents, owner);
        
        agent.name = name;
        agent.description = description;
        agent.endpoint_url = endpoint_url;
        agent.is_active = is_active;
        agent.updated_at = timestamp::now_seconds();

        event::emit(AgentUpdated {
            owner,
            name,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// View functions

    #[view]
    public fun get_agent_info(owner: address): AgentInfo acquires AgentRegistry {
        let registry = borrow_global<AgentRegistry>(@agent_registry);
        assert!(table::contains(&registry.agents, owner), E_AGENT_NOT_FOUND);
        *table::borrow(&registry.agents, owner)
    }

    #[view]
    public fun get_agent_rating(owner: address): u64 acquires AgentRegistry {
        let registry = borrow_global<AgentRegistry>(@agent_registry);
        assert!(table::contains(&registry.agents, owner), E_AGENT_NOT_FOUND);
        
        let agent = table::borrow(&registry.agents, owner);
        if (agent.rating_count == 0) {
            0
        } else {
            agent.rating_sum / agent.rating_count
        }
    }

    #[view]
    public fun get_capability_price(owner: address, capability_name: String): u64 acquires AgentRegistry {
        let registry = borrow_global<AgentRegistry>(@agent_registry);
        assert!(table::contains(&registry.agents, owner), E_AGENT_NOT_FOUND);
        
        let agent = table::borrow(&registry.agents, owner);
        let i = 0;
        let len = vector::length(&agent.capabilities);
        
        while (i < len) {
            let cap = vector::borrow(&agent.capabilities, i);
            if (cap.name == capability_name) {
                return cap.price
            };
            i = i + 1;
        };
        
        0
    }

    #[view]
    public fun get_agent_count(): u64 acquires AgentRegistry {
        let registry = borrow_global<AgentRegistry>(@agent_registry);
        registry.agent_count
    }

    #[view]
    public fun is_agent_registered(owner: address): bool acquires AgentRegistry {
        let registry = borrow_global<AgentRegistry>(@agent_registry);
        table::contains(&registry.agents, owner)
    }
}