module agent_registry::agent_marketplace {
    use std::string::{String, utf8};
    use std::signer;
    use std::vector;
    use aptos_std::table::{Self, Table};
    use aptos_std::simple_map::{Self, SimpleMap};
    use aptos_framework::timestamp;
    use aptos_framework::event;
    use aptos_framework::account;

    /// Error codes
    const E_NOT_AUTHORIZED: u64 = 1;
    const E_AGENT_NOT_FOUND: u64 = 2;
    const E_TASK_NOT_FOUND: u64 = 3;
    const E_INSUFFICIENT_PAYMENT: u64 = 4;
    const E_TASK_ALREADY_ASSIGNED: u64 = 5;
    const E_INVALID_STATE: u64 = 6;
    const E_DISPUTE_ALREADY_EXISTS: u64 = 7;

    /// Task status
    const TASK_STATUS_OPEN: u8 = 0;
    const TASK_STATUS_ASSIGNED: u8 = 1;
    const TASK_STATUS_COMPLETED: u8 = 2;
    const TASK_STATUS_DISPUTED: u8 = 3;
    const TASK_STATUS_CANCELLED: u8 = 4;

    /// Agent profile with on-chain identity
    struct AgentProfile has store, copy, drop {
        name: String,
        description: String,
        capabilities: vector<String>,
        endpoint_url: String,
        owner: address,
        created_at: u64,
        verification_status: u8, // 0=unverified, 1=pending, 2=verified
    }

    /// Agent reputation metrics
    struct AgentReputation has store, copy, drop {
        // Trust metrics
        total_tasks_completed: u64,
        total_tasks_failed: u64,
        total_disputes: u64,
        disputes_won: u64,
        
        // Quality metrics
        total_ratings: u64,
        rating_sum: u64,
        average_rating: u64, // scaled by 100 (e.g., 450 = 4.5/5)
        
        // Payment metrics
        total_earned: u64,
        total_paid: u64,
        successful_payments: u64,
        failed_payments: u64,
        
        // Timeliness
        on_time_deliveries: u64,
        late_deliveries: u64,
        
        // Last updated
        updated_at: u64,
    }

    /// Task structure - micro-task marketplace
    struct Task has store, copy, drop {
        task_id: u64,
        creator: address,
        assigned_agent: address,
        title: String,
        description: String,
        requirements: vector<String>,
        budget: u64,
        deadline: u64,
        status: u8,
        created_at: u64,
        completed_at: u64,
        result_cid: String, // IPFS hash of result
        dispute_reason: String,
    }

    /// Payment record for trust building
    struct PaymentRecord has store, copy, drop {
        payment_id: u64,
        from: address,
        to: address,
        amount: u64,
        task_id: u64,
        success: bool,
        timestamp: u64,
        tx_hash: String,
    }

    /// Agent behavior/preferences storage
    struct AgentBehavior has store {
        // Learning preferences
        preferred_task_types: vector<String>,
        preferred_budget_range: vector<u64>, // [min, max]
        
        // Performance history (last 100 tasks)
        recent_task_ids: vector<u64>,
        
        // Client preferences
        preferred_clients: vector<address>,
        blocked_clients: vector<address>,
        
        // Custom preferences (key-value storage)
        preferences: SimpleMap<String, String>,
        
        // Skills with proficiency (0-100)
        skills: SimpleMap<String, u64>,
        
        updated_at: u64,
    }

    /// Main marketplace state
    struct AgentMarketplace has key {
        // Registered agents
        agents: Table<address, AgentProfile>,
        
        // Agent reputations
        reputations: Table<address, AgentReputation>,
        
        // Agent behaviors/preferences
        behaviors: Table<address, AgentBehavior>,
        
        // All tasks
        tasks: Table<u64, Task>,
        
        // Payment history
        payments: Table<u64, PaymentRecord>,
        
        // Agent's tasks (agent -> task_ids)
        agent_tasks: Table<address, vector<u64>>,
        
        // Creator's tasks (creator -> task_ids)
        creator_tasks: Table<address, vector<u64>>,
        
        // Counters
        next_task_id: u64,
        next_payment_id: u64,
        agent_count: u64,
        
        // Admin
        admin: address,
    }

    /// Events
    #[event]
    struct AgentRegistered has drop, store {
        agent: address,
        name: String,
        timestamp: u64,
    }

    #[event]
    struct TaskCreated has drop, store {
        task_id: u64,
        creator: address,
        budget: u64,
        timestamp: u64,
    }

    #[event]
    struct TaskAssigned has drop, store {
        task_id: u64,
        agent: address,
        timestamp: u64,
    }

    #[event]
    struct TaskCompleted has drop, store {
        task_id: u64,
        agent: address,
        payment: u64,
        timestamp: u64,
    }

    #[event]
    struct PaymentRecorded has drop, store {
        payment_id: u64,
        from: address,
        to: address,
        amount: u64,
        success: bool,
        timestamp: u64,
    }

    #[event]
    struct ReputationUpdated has drop, store {
        agent: address,
        new_rating: u64,
        tasks_completed: u64,
        timestamp: u64,
    }

    #[event]
    struct DisputeRaised has drop, store {
        task_id: u64,
        reason: String,
        timestamp: u64,
    }

    /// Initialize marketplace
    fun init_module(account: &signer) {
        let admin = signer::address_of(account);
        
        move_to(account, AgentMarketplace {
            agents: table::new(),
            reputations: table::new(),
            behaviors: table::new(),
            tasks: table::new(),
            payments: table::new(),
            agent_tasks: table::new(),
            creator_tasks: table::new(),
            next_task_id: 1,
            next_payment_id: 1,
            agent_count: 0,
            admin,
        });
    }

    /// Register as an agent
    public entry fun register_agent(
        account: &signer,
        name: String,
        description: String,
        capabilities: vector<String>,
        endpoint_url: String,
    ) acquires AgentMarketplace {
        let agent_addr = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        assert!(!table::contains(&marketplace.agents, agent_addr), E_NOT_AUTHORIZED);

        // Create profile
        let profile = AgentProfile {
            name,
            description,
            capabilities,
            endpoint_url,
            owner: agent_addr,
            created_at: timestamp::now_seconds(),
            verification_status: 0,
        };

        // Initialize reputation
        let reputation = AgentReputation {
            total_tasks_completed: 0,
            total_tasks_failed: 0,
            total_disputes: 0,
            disputes_won: 0,
            total_ratings: 0,
            rating_sum: 0,
            average_rating: 0,
            total_earned: 0,
            total_paid: 0,
            successful_payments: 0,
            failed_payments: 0,
            on_time_deliveries: 0,
            late_deliveries: 0,
            updated_at: timestamp::now_seconds(),
        };

        // Initialize behavior
        let behavior = AgentBehavior {
            preferred_task_types: vector::empty(),
            preferred_budget_range: vector::empty(),
            recent_task_ids: vector::empty(),
            preferred_clients: vector::empty(),
            blocked_clients: vector::empty(),
            preferences: simple_map::create(),
            skills: simple_map::create(),
            updated_at: timestamp::now_seconds(),
        };

        table::add(&mut marketplace.agents, agent_addr, profile);
        table::add(&mut marketplace.reputations, agent_addr, reputation);
        table::add(&mut marketplace.behaviors, agent_addr, behavior);
        table::add(&mut marketplace.agent_tasks, agent_addr, vector::empty());
        
        marketplace.agent_count = marketplace.agent_count + 1;

        event::emit(AgentRegistered {
            agent: agent_addr,
            name,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Create a new task (micro-task marketplace)
    public entry fun create_task(
        account: &signer,
        title: String,
        description: String,
        requirements: vector<String>,
        budget: u64,
        deadline: u64,
    ) acquires AgentMarketplace {
        let creator = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        let task_id = marketplace.next_task_id;
        
        let task = Task {
            task_id,
            creator,
            assigned_agent: @0x0,
            title,
            description,
            requirements,
            budget,
            deadline,
            status: TASK_STATUS_OPEN,
            created_at: timestamp::now_seconds(),
            completed_at: 0,
            result_cid: utf8(b""),
            dispute_reason: utf8(b""),
        };

        table::add(&mut marketplace.tasks, task_id, task);
        
        // Add to creator's tasks
        if (!table::contains(&marketplace.creator_tasks, creator)) {
            table::add(&mut marketplace.creator_tasks, creator, vector::empty());
        };
        let creator_task_list = table::borrow_mut(&mut marketplace.creator_tasks, creator);
        vector::push_back(creator_task_list, task_id);

        marketplace.next_task_id = task_id + 1;

        event::emit(TaskCreated {
            task_id,
            creator,
            budget,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Accept/assign a task
    public entry fun accept_task(
        account: &signer,
        task_id: u64,
    ) acquires AgentMarketplace {
        let agent_addr = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        assert!(table::contains(&marketplace.agents, agent_addr), E_AGENT_NOT_FOUND);
        assert!(table::contains(&marketplace.tasks, task_id), E_TASK_NOT_FOUND);
        
        let task = table::borrow_mut(&mut marketplace.tasks, task_id);
        assert!(task.status == TASK_STATUS_OPEN, E_TASK_ALREADY_ASSIGNED);
        
        task.assigned_agent = agent_addr;
        task.status = TASK_STATUS_ASSIGNED;
        
        // Add to agent's tasks
        let agent_task_list = table::borrow_mut(&mut marketplace.agent_tasks, agent_addr);
        vector::push_back(agent_task_list, task_id);

        event::emit(TaskAssigned {
            task_id,
            agent: agent_addr,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Complete a task and record payment
    public entry fun complete_task(
        account: &signer,
        task_id: u64,
        result_cid: String,
        on_time: bool,
    ) acquires AgentMarketplace {
        let agent_addr = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        assert!(table::contains(&marketplace.tasks, task_id), E_TASK_NOT_FOUND);
        
        let task = table::borrow_mut(&mut marketplace.tasks, task_id);
        assert!(task.assigned_agent == agent_addr, E_NOT_AUTHORIZED);
        assert!(task.status == TASK_STATUS_ASSIGNED, E_INVALID_STATE);
        
        task.status = TASK_STATUS_COMPLETED;
        task.completed_at = timestamp::now_seconds();
        task.result_cid = result_cid;
        
        // Update agent reputation
        let reputation = table::borrow_mut(&mut marketplace.reputations, agent_addr);
        reputation.total_tasks_completed = reputation.total_tasks_completed + 1;
        reputation.total_earned = reputation.total_earned + task.budget;
        
        if (on_time) {
            reputation.on_time_deliveries = reputation.on_time_deliveries + 1;
        } else {
            reputation.late_deliveries = reputation.late_deliveries + 1;
        };
        
        reputation.updated_at = timestamp::now_seconds();

        event::emit(TaskCompleted {
            task_id,
            agent: agent_addr,
            payment: task.budget,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Record a payment (called after on-chain payment)
    public entry fun record_payment(
        account: &signer,
        to: address,
        amount: u64,
        task_id: u64,
        success: bool,
        tx_hash: String,
    ) acquires AgentMarketplace {
        let from = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        let payment_id = marketplace.next_payment_id;
        
        let payment = PaymentRecord {
            payment_id,
            from,
            to,
            amount,
            task_id,
            success,
            timestamp: timestamp::now_seconds(),
            tx_hash,
        };

        table::add(&mut marketplace.payments, payment_id, payment);
        marketplace.next_payment_id = payment_id + 1;
        
        // Update reputation
        if (table::contains(&marketplace.reputations, to)) {
            let reputation = table::borrow_mut(&mut marketplace.reputations, to);
            if (success) {
                reputation.successful_payments = reputation.successful_payments + 1;
            } else {
                reputation.failed_payments = reputation.failed_payments + 1;
            };
        };

        event::emit(PaymentRecorded {
            payment_id,
            from,
            to,
            amount,
            success,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Rate an agent after task completion
    public entry fun rate_agent(
        account: &signer,
        agent: address,
        task_id: u64,
        rating: u64, // 1-500 (1.0-5.0 stars, scaled by 100)
    ) acquires AgentMarketplace {
        let rater = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        assert!(table::contains(&marketplace.tasks, task_id), E_TASK_NOT_FOUND);
        let task = table::borrow(&marketplace.tasks, task_id);
        
        // Only task creator can rate
        assert!(task.creator == rater, E_NOT_AUTHORIZED);
        assert!(task.assigned_agent == agent, E_NOT_AUTHORIZED);
        assert!(task.status == TASK_STATUS_COMPLETED, E_INVALID_STATE);
        
        let reputation = table::borrow_mut(&mut marketplace.reputations, agent);
        reputation.total_ratings = reputation.total_ratings + 1;
        reputation.rating_sum = reputation.rating_sum + rating;
        reputation.average_rating = reputation.rating_sum / reputation.total_ratings;
        reputation.updated_at = timestamp::now_seconds();

        event::emit(ReputationUpdated {
            agent,
            new_rating: reputation.average_rating,
            tasks_completed: reputation.total_tasks_completed,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Raise a dispute
    public entry fun raise_dispute(
        account: &signer,
        task_id: u64,
        reason: String,
    ) acquires AgentMarketplace {
        let creator = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        assert!(table::contains(&marketplace.tasks, task_id), E_TASK_NOT_FOUND);
        let task = table::borrow_mut(&mut marketplace.tasks, task_id);
        
        assert!(task.creator == creator, E_NOT_AUTHORIZED);
        assert!(task.status != TASK_STATUS_DISPUTED, E_DISPUTE_ALREADY_EXISTS);
        
        task.status = TASK_STATUS_DISPUTED;
        task.dispute_reason = reason;
        
        // Update agent's dispute count
        if (table::contains(&marketplace.reputations, task.assigned_agent)) {
            let reputation = table::borrow_mut(&mut marketplace.reputations, task.assigned_agent);
            reputation.total_disputes = reputation.total_disputes + 1;
        };

        event::emit(DisputeRaised {
            task_id,
            reason,
            timestamp: timestamp::now_seconds(),
        });
    }

    /// Update agent behavior/preferences
    public entry fun update_preferences(
        account: &signer,
        key: String,
        value: String,
    ) acquires AgentMarketplace {
        let agent = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        assert!(table::contains(&marketplace.behaviors, agent), E_AGENT_NOT_FOUND);
        
        let behavior = table::borrow_mut(&mut marketplace.behaviors, agent);
        simple_map::upsert(&mut behavior.preferences, key, value);
        behavior.updated_at = timestamp::now_seconds();
    }

    /// Update skills
    public entry fun update_skill(
        account: &signer,
        skill: String,
        proficiency: u64, // 0-100
    ) acquires AgentMarketplace {
        let agent = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        assert!(table::contains(&marketplace.behaviors, agent), E_AGENT_NOT_FOUND);
        
        let behavior = table::borrow_mut(&mut marketplace.behaviors, agent);
        simple_map::upsert(&mut behavior.skills, skill, proficiency);
        behavior.updated_at = timestamp::now_seconds();
    }

    /// Block/unblock a client
    public entry fun block_client(
        account: &signer,
        client: address,
        block: bool,
    ) acquires AgentMarketplace {
        let agent = signer::address_of(account);
        let marketplace = borrow_global_mut<AgentMarketplace>(@agent_registry);
        
        assert!(table::contains(&marketplace.behaviors, agent), E_AGENT_NOT_FOUND);
        
        let behavior = table::borrow_mut(&mut marketplace.behaviors, agent);
        
        if (block) {
            vector::push_back(&mut behavior.blocked_clients, client);
        } else {
            // Remove from blocked list (simplified - would need find/remove)
        };
        behavior.updated_at = timestamp::now_seconds();
    }

    // ============== VIEW FUNCTIONS ==============

    #[view]
    public fun get_agent_profile(agent: address): AgentProfile acquires AgentMarketplace {
        let marketplace = borrow_global<AgentMarketplace>(@agent_registry);
        assert!(table::contains(&marketplace.agents, agent), E_AGENT_NOT_FOUND);
        *table::borrow(&marketplace.agents, agent)
    }

    #[view]
    public fun get_agent_reputation(agent: address): AgentReputation acquires AgentMarketplace {
        let marketplace = borrow_global<AgentMarketplace>(@agent_registry);
        assert!(table::contains(&marketplace.reputations, agent), E_AGENT_NOT_FOUND);
        *table::borrow(&marketplace.reputations, agent)
    }

    #[view]
    public fun get_trust_score(agent: address): u64 acquires AgentMarketplace {
        let marketplace = borrow_global<AgentMarketplace>(@agent_registry);
        assert!(table::contains(&marketplace.reputations, agent), E_AGENT_NOT_FOUND);
        
        let rep = table::borrow(&marketplace.reputations, agent);
        
        // Calculate trust score (0-10000, scaled by 100)
        if (rep.total_tasks_completed == 0) {
            return 5000 // Neutral score for new agents
        };
        
        let completion_rate = (rep.total_tasks_completed * 10000) / 
            (rep.total_tasks_completed + rep.total_tasks_failed);
        
        let dispute_rate = if (rep.total_tasks_completed > 0) {
            (rep.total_disputes * 10000) / rep.total_tasks_completed
        } else { 0 };
        
        let payment_success_rate = if (rep.successful_payments + rep.failed_payments > 0) {
            (rep.successful_payments * 10000) / 
            (rep.successful_payments + rep.failed_payments)
        } else { 10000 };
        
        let on_time_rate = if (rep.on_time_deliveries + rep.late_deliveries > 0) {
            (rep.on_time_deliveries * 10000) / 
            (rep.on_time_deliveries + rep.late_deliveries)
        } else { 10000 };
        
        // Weighted combination
        (completion_rate * 30 + payment_success_rate * 30 + on_time_rate * 25 + 
         (10000 - dispute_rate) * 15) / 100
    }

    #[view]
    public fun get_task(task_id: u64): Task acquires AgentMarketplace {
        let marketplace = borrow_global<AgentMarketplace>(@agent_registry);
        assert!(table::contains(&marketplace.tasks, task_id), E_TASK_NOT_FOUND);
        *table::borrow(&marketplace.tasks, task_id)
    }

    #[view]
    public fun get_agent_tasks(agent: address): vector<u64> acquires AgentMarketplace {
        let marketplace = borrow_global<AgentMarketplace>(@agent_registry);
        if (table::contains(&marketplace.agent_tasks, agent)) {
            *table::borrow(&marketplace.agent_tasks, agent)
        } else {
            vector::empty()
        }
    }

    #[view]
    public fun is_agent_verified(agent: address): bool acquires AgentMarketplace {
        let marketplace = borrow_global<AgentMarketplace>(@agent_registry);
        if (table::contains(&marketplace.agents, agent)) {
            let profile = table::borrow(&marketplace.agents, agent);
            profile.verification_status == 2
        } else {
            false
        }
    }

    #[view]
    public fun get_payment_history(agent: address): vector<PaymentRecord> acquires AgentMarketplace {
        let marketplace = borrow_global<AgentMarketplace>(@agent_registry);
        let history = vector::empty<PaymentRecord>();
        
        let i = 1;
        while (i < marketplace.next_payment_id) {
            if (table::contains(&marketplace.payments, i)) {
                let payment = table::borrow(&marketplace.payments, i);
                if (payment.to == agent || payment.from == agent) {
                    vector::push_back(&mut history, *payment);
                }
            };
            i = i + 1;
        };
        
        history
    }
}