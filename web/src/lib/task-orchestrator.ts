/**
 * Task Orchestrator - "Uber for AI Agents"
 * 
 * Breaks large tasks into smaller micro-tasks and coordinates
 * multiple specialist agents to complete them.
 * 
 * ## How It Works
 * 
 * ### 1. Task Decomposition
 * ```
 * Complex Task: "Analyze tech sector"
 *    ↓
 * Decomposed into:
 * ├── Task 1: Analyze AAPL ($0.0001) → Agent A (momentum specialist)
 * ├── Task 2: Analyze MSFT ($0.0001) → Agent B (fundamentals specialist)
 * ├── Task 3: Analyze GOOGL ($0.0001) → Agent C (sentiment specialist)
 * └── Task 4: Aggregate results ($0.0002) → Agent D (aggregator)
 * ```
 * 
 * ### 2. Agent Discovery & Selection
 * ```
 * Find agents by:
 * - Capabilities (momentum, fundamentals, sentiment)
 * - Reputation score (> 80/100)
 * - Price range (competitive)
 * - Availability (online status)
 * ```
 * 
 * ### 3. Coordination
 * ```
 * Orchestrator manages:
 * - Task assignment
 * - Payment escrow
 * - Result aggregation
 * - Failure handling (retry with other agents)
 * ```
 * 
 * ### 4. Payment Flow
 * ```
 * Client pays Orchestrator (total budget)
 *    ↓
 * Orchestrator holds in escrow
 *    ↓
 * Each sub-task payment released on completion
 *    ↓
 * Unused funds returned to client
 * ```
 * 
 * ## Example: Multi-Agent Stock Analysis
 * 
 * ```typescript
 * const orchestrator = new TaskOrchestrator(xmtpClient)
 * 
 * const result = await orchestrator.executeComplexTask({
 *   type: 'comprehensive-analysis',
 *   target: 'AAPL',
 *   budget: 1000, // octas
 *   maxAgents: 3,
 *   decomposition: [
 *     { type: 'momentum', weight: 0.4 },
 *     { type: 'fundamentals', weight: 0.4 },
 *     { type: 'sentiment', weight: 0.2 }
 *   ]
 * })
 * ```
 */

import { XMTPAgentClient, AgentMessage } from './xmtp-client'
import { x402Client } from './x402-client'

export interface SubTask {
  id: string
  type: string
  description: string
  budget: number
  assignedAgent?: AgentInfo
  status: 'pending' | 'assigned' | 'completed' | 'failed'
  result?: any
  deadline: number
}

export interface AgentInfo {
  address: string
  endpoint: string
  capabilities: string[]
  reputation: number
  pricePerRequest: number
  availability: 'online' | 'busy' | 'offline'
}

export interface ComplexTask {
  id: string
  type: string
  description: string
  totalBudget: number
  subTasks: SubTask[]
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  createdAt: number
  deadline: number
  results?: any
}

export interface OrchestratorConfig {
  minReputation: number
  maxRetries: number
  escrowEnabled: boolean
  xmtp?: XMTPAgentClient
}

/**
 * Task Orchestrator
 * 
 * The "Uber for AI Agents" - coordinates multiple specialist agents
 * to complete complex tasks through micro-task decomposition.
 */
export class TaskOrchestrator {
  private xmtp?: XMTPAgentClient
  private config: OrchestratorConfig
  private activeTasks: Map<string, ComplexTask> = new Map()
  private agentRegistry: Map<string, AgentInfo> = new Map()

  constructor(config: Partial<OrchestratorConfig> = {}) {
    this.config = {
      minReputation: 70,
      maxRetries: 2,
      escrowEnabled: true,
      ...config,
    }
    this.xmtp = config.xmtp
  }

  /**
   * Register an agent in the orchestrator's registry
   */
  registerAgent(agent: AgentInfo): void {
    this.agentRegistry.set(agent.address, agent)
    console.log(`Agent registered: ${agent.address.slice(0, 10)}... (${agent.capabilities.join(', ')})`)
  }

  /**
   * Find the best agent for a specific task type
   */
  findBestAgent(taskType: string, budget: number): AgentInfo | null {
    const candidates: AgentInfo[] = []

    for (const agent of this.agentRegistry.values()) {
      // Check capability match
      if (agent.capabilities.includes(taskType)) {
        // Check reputation threshold
        if (agent.reputation >= this.config.minReputation) {
          // Check price is within budget
          if (agent.pricePerRequest <= budget) {
            // Check availability
            if (agent.availability === 'online') {
              candidates.push(agent)
            }
          }
        }
      }
    }

    if (candidates.length === 0) {
      console.log(`No suitable agent found for ${taskType}`)
      return null
    }

    // Sort by: reputation (desc), then price (asc)
    candidates.sort((a, b) => {
      if (b.reputation !== a.reputation) {
        return b.reputation - a.reputation
      }
      return a.pricePerRequest - b.pricePerRequest
    })

    return candidates[0]
  }

  /**
   * Decompose a complex task into micro-tasks
   * 
   * @example
   * ```typescript
   * const subTasks = orchestrator.decomposeTask({
   *   type: 'comprehensive-analysis',
   *   target: 'AAPL',
   *   budget: 1000
   * })
   * // Returns: [momentum task, fundamentals task, sentiment task, aggregation task]
   * ```
   */
  decomposeTask(
    taskType: string,
    target: string,
    totalBudget: number
  ): SubTask[] {
    const decompositionStrategies: Record<string, () => SubTask[]> = {
      'comprehensive-analysis': () => [
        {
          id: `momentum-${Date.now()}`,
          type: 'momentum',
          description: `Analyze momentum indicators for ${target}`,
          budget: Math.floor(totalBudget * 0.25),
          status: 'pending',
          deadline: Date.now() + 300000, // 5 minutes
        },
        {
          id: `fundamentals-${Date.now()}`,
          type: 'fundamentals',
          description: `Analyze fundamentals for ${target}`,
          budget: Math.floor(totalBudget * 0.25),
          status: 'pending',
          deadline: Date.now() + 300000,
        },
        {
          id: `sentiment-${Date.now()}`,
          type: 'sentiment',
          description: `Analyze social sentiment for ${target}`,
          budget: Math.floor(totalBudget * 0.20),
          status: 'pending',
          deadline: Date.now() + 300000,
        },
        {
          id: `aggregate-${Date.now()}`,
          type: 'aggregate',
          description: `Aggregate all analyses for ${target}`,
          budget: Math.floor(totalBudget * 0.30),
          status: 'pending',
          deadline: Date.now() + 360000, // 6 minutes
        },
      ],
      
      'sector-analysis': () => [
        {
          id: `sector-scan-${Date.now()}`,
          type: 'sector-scan',
          description: `Scan ${target} sector for top performers`,
          budget: Math.floor(totalBudget * 0.40),
          status: 'pending',
          deadline: Date.now() + 600000,
        },
        {
          id: `sector-leaders-${Date.now()}`,
          type: 'analyze-multiple',
          description: `Analyze top 3 performers in ${target}`,
          budget: Math.floor(totalBudget * 0.40),
          status: 'pending',
          deadline: Date.now() + 600000,
        },
        {
          id: `sector-report-${Date.now()}`,
          type: 'report',
          description: `Generate ${target} sector report`,
          budget: Math.floor(totalBudget * 0.20),
          status: 'pending',
          deadline: Date.now() + 660000,
        },
      ],
      
      'portfolio-scan': () => [
        {
          id: `portfolio-health-${Date.now()}`,
          type: 'health-check',
          description: `Health check for portfolio ${target}`,
          budget: Math.floor(totalBudget * 0.30),
          status: 'pending',
          deadline: Date.now() + 300000,
        },
        {
          id: `risk-analysis-${Date.now()}`,
          type: 'risk',
          description: `Risk analysis for portfolio ${target}`,
          budget: Math.floor(totalBudget * 0.35),
          status: 'pending',
          deadline: Date.now() + 300000,
        },
        {
          id: `rebalancing-${Date.now()}`,
          type: 'rebalance',
          description: `Rebalancing suggestions for ${target}`,
          budget: Math.floor(totalBudget * 0.35),
          status: 'pending',
          deadline: Date.now() + 360000,
        },
      ],
    }

    const strategy = decompositionStrategies[taskType]
    if (!strategy) {
      // Default: single task
      return [{
        id: `task-${Date.now()}`,
        type: taskType,
        description: `Execute ${taskType} for ${target}`,
        budget: totalBudget,
        status: 'pending',
        deadline: Date.now() + 300000,
      }]
    }

    return strategy()
  }

  /**
   * Execute a complex task with agent coordination
   */
  async executeComplexTask(params: {
    type: string
    target: string
    budget: number
    maxAgents?: number
  }): Promise<ComplexTask> {
    const { type, target, budget, maxAgents = 5 } = params

    // 1. Decompose task
    console.log(`Decomposing ${type} task for ${target}...`)
    const subTasks = this.decomposeTask(type, target, budget)
    console.log(`Decomposed into ${subTasks.length} sub-tasks`)

    // 2. Create complex task
    const complexTask: ComplexTask = {
      id: `complex-${Date.now()}`,
      type,
      description: `${type} for ${target}`,
      totalBudget: budget,
      subTasks,
      status: 'in_progress',
      createdAt: Date.now(),
      deadline: Date.now() + 600000, // 10 minutes
    }

    this.activeTasks.set(complexTask.id, complexTask)

    // 3. Assign agents to sub-tasks
    for (const subTask of subTasks) {
      const agent = this.findBestAgent(subTask.type, subTask.budget)
      
      if (agent) {
        subTask.assignedAgent = agent
        subTask.status = 'assigned'
        console.log(`Assigned ${subTask.type} to ${agent.address.slice(0, 10)}...`)
        
        // Send XMTP notification if available
        if (this.xmtp) {
          try {
            await this.xmtp.sendTaskRequest(agent.address, {
              taskType: subTask.type,
              description: subTask.description,
              budget: subTask.budget,
              deadline: subTask.deadline,
              requirements: [target],
            })
          } catch (err) {
            console.warn(`XMTP notification failed for ${subTask.type}:`, err)
          }
        }
      } else {
        console.warn(`No agent found for ${subTask.type}`)
        subTask.status = 'failed'
      }
    }

    // 4. Execute sub-tasks (in parallel where possible)
    await this.executeSubTasks(complexTask)

    // 5. Aggregate results
    if (this.allSubTasksCompleted(complexTask)) {
      complexTask.results = this.aggregateResults(complexTask)
      complexTask.status = 'completed'
    } else if (this.someSubTasksFailed(complexTask)) {
      complexTask.status = 'failed'
    }

    return complexTask
  }

  /**
   * Execute all assigned sub-tasks
   */
  private async executeSubTasks(complexTask: ComplexTask): Promise<void> {
    const promises = complexTask.subTasks
      .filter(st => st.status === 'assigned' && st.assignedAgent)
      .map(async (subTask) => {
        try {
          console.log(`Executing sub-task: ${subTask.type}`)
          
          // Call agent's API with x402 payment
          const agent = subTask.assignedAgent!
          const client = x402Client({
            privateKey: process.env.ORCHESTRATOR_PRIVATE_KEY || '',
            baseUrl: agent.endpoint,
            gasless: true,
          })

          const result = await client.get(
            `/api/agent/${subTask.type}?symbol=${subTask.description.split('for ')[1]}`
          )

          subTask.result = result
          subTask.status = 'completed'
          console.log(`✅ Sub-task completed: ${subTask.type}`)
          
        } catch (error) {
          console.error(`❌ Sub-task failed: ${subTask.type}`, error)
          subTask.status = 'failed'
          
          // Retry with another agent if retries available
          await this.retrySubTask(complexTask, subTask)
        }
      })

    await Promise.all(promises)
  }

  /**
   * Retry a failed sub-task with another agent
   */
  private async retrySubTask(
    complexTask: ComplexTask,
    failedSubTask: SubTask
  ): Promise<void> {
    let retries = 0
    
    while (retries < this.config.maxRetries && failedSubTask.status === 'failed') {
      console.log(`Retrying ${failedSubTask.type} (attempt ${retries + 1})...`)
      
      // Find alternative agent
      const alternativeAgent = this.findAlternativeAgent(
        failedSubTask.type,
        failedSubTask.budget,
        failedSubTask.assignedAgent?.address
      )
      
      if (alternativeAgent) {
        failedSubTask.assignedAgent = alternativeAgent
        failedSubTask.status = 'assigned'
        
        try {
          const client = x402Client({
            privateKey: process.env.ORCHESTRATOR_PRIVATE_KEY || '',
            baseUrl: alternativeAgent.endpoint,
            gasless: true,
          })

          const result = await client.get(
            `/api/agent/${failedSubTask.type}?symbol=${failedSubTask.description.split('for ')[1]}`
          )

          failedSubTask.result = result
          failedSubTask.status = 'completed'
          console.log(`✅ Retry successful: ${failedSubTask.type}`)
          return
          
        } catch (error) {
          console.error(`❌ Retry failed: ${failedSubTask.type}`)
          retries++
        }
      } else {
        console.error(`No alternative agent for ${failedSubTask.type}`)
        break
      }
    }
  }

  /**
   * Find an alternative agent (exclude the failed one)
   */
  private findAlternativeAgent(
    taskType: string,
    budget: number,
    excludeAddress?: string
  ): AgentInfo | null {
    const candidates: AgentInfo[] = []

    for (const [address, agent] of this.agentRegistry) {
      if (address === excludeAddress) continue
      if (agent.capabilities.includes(taskType) && agent.pricePerRequest <= budget) {
        candidates.push(agent)
      }
    }

    candidates.sort((a, b) => b.reputation - a.reputation)
    return candidates[0] || null
  }

  /**
   * Aggregate results from all sub-tasks
   */
  private aggregateResults(complexTask: ComplexTask): any {
    const results: Record<string, any> = {}
    
    for (const subTask of complexTask.subTasks) {
      if (subTask.result) {
        results[subTask.type] = subTask.result
      }
    }

    // Weighted aggregation based on task type
    if (complexTask.type === 'comprehensive-analysis') {
      return this.aggregateComprehensiveAnalysis(results)
    }

    return results
  }

  /**
   * Aggregate comprehensive analysis results
   */
  private aggregateResultsFromSubtasks(results: Record<string, any>): any {
    const momentum = results['momentum']?.momentum_score || 50
    const fundamentals = results['fundamentals']?.value_score || 50
    const sentiment = results['sentiment']?.sentiment_score || 50

    // Weighted average
    const overallScore = (momentum * 0.4 + fundamentals * 0.4 + sentiment * 0.2)

    let signal = 'hold'
    if (overallScore >= 70) signal = 'strong_buy'
    else if (overallScore >= 55) signal = 'buy'
    else if (overallScore <= 30) signal = 'strong_sell'
    else if (overallScore <= 45) signal = 'sell'

    return {
      overall_score: Math.round(overallScore),
      signal,
      components: results,
      aggregated_at: Date.now(),
    }
  }

  private allSubTasksCompleted(task: ComplexTask): boolean {
    return task.subTasks.every(st => st.status === 'completed')
  }

  private someSubTasksFailed(task: ComplexTask): boolean {
    return task.subTasks.some(st => st.status === 'failed')
  }

  private aggregateComprehensiveAnalysis(results: Record<string, any>): any {
    return this.aggregateResultsFromSubtasks(results)
  }

  /**
   * Get task status
   */
  getTaskStatus(taskId: string): ComplexTask | undefined {
    return this.activeTasks.get(taskId)
  }

  /**
   * List all active tasks
   */
  getActiveTasks(): ComplexTask[] {
    return Array.from(this.activeTasks.values())
  }
}

// Export singleton
export const taskOrchestrator = new TaskOrchestrator()
