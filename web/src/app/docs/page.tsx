"use client"

import { useEffect, useState } from "react"
import {
    ChevronRight,
    ChevronDown,
    Code2,
    Cpu,
    Database,
    Globe,
    Layers,
    Shield,
    Zap,
    Brain,
    Activity,
    ArrowRight,
    Terminal,
    GitBranch,
    Server,
    TrendingUp,
    BarChart3,
    Search,
    Settings,
    Lock,
    LayoutGrid,
    Workflow,
    FileText,
    ExternalLink,
    Copy,
    Check,
} from "lucide-react"

/* ─────────── types ─────────── */
interface TocItem {
    id: string
    label: string
    icon: React.ElementType
    children?: { id: string; label: string }[]
}

/* ─────────── table-of-contents data ─────────── */
const toc: TocItem[] = [
    {
        id: "overview",
        label: "What Is Stock Radar?",
        icon: Zap,
    },
    {
        id: "architecture",
        label: "System Architecture",
        icon: Layers,
        children: [
            { id: "architecture-diagram", label: "Architecture Diagram" },
            { id: "design-philosophy", label: "Design Philosophy" },
            { id: "runtime-flow", label: "Runtime Request Flows" },
        ],
    },
    {
        id: "orchestrator",
        label: "The Orchestrator",
        icon: Workflow,
        children: [
            { id: "orchestrator-init", label: "Initialization" },
            { id: "analysis-pipeline", label: "5-Step Pipeline" },
        ],
    },
    {
        id: "agents",
        label: "Agent Layer",
        icon: Cpu,
        children: [
            { id: "fetcher", label: "StockFetcher" },
            { id: "analyzer", label: "StockAnalyzer" },
            { id: "storage", label: "StockStorage" },
            { id: "search-types", label: "Search Types" },
            { id: "notifications", label: "NotificationManager" },
            { id: "scorer", label: "StockScorer" },
            { id: "chat", label: "ChatAssistant" },
            { id: "rag", label: "RAG System" },
            { id: "realtime", label: "RealtimeManager" },
            { id: "usage-tracker", label: "UsageTracker" },
        ],
    },
    {
        id: "ml",
        label: "ML & Training",
        icon: Brain,
        children: [
            { id: "predictor", label: "SignalPredictor" },
            { id: "paper-trading", label: "Paper Trading" },
            { id: "broker", label: "Broker Adapter" },
            { id: "risk-mgmt", label: "Risk Management" },
        ],
    },
    {
        id: "infra",
        label: "Infrastructure",
        icon: Settings,
        children: [
            { id: "config", label: "Configuration" },
            { id: "config-reality", label: "Current Runtime Reality" },
            { id: "caching", label: "Caching Layer" },
            { id: "guardrails", label: "LLM Guardrails" },
            { id: "prompts", label: "Prompt Versioning" },
            { id: "streaming", label: "SSE Streaming" },
            { id: "metrics-infra", label: "Prometheus Metrics" },
            { id: "token-accounting", label: "Token Accounting" },
        ],
    },
    {
        id: "api",
        label: "Backend API",
        icon: Server,
    },
    {
        id: "pipeline-walkthrough",
        label: "Full Pipeline Walkthrough",
        icon: GitBranch,
    },
    {
        id: "cli",
        label: "CLI Reference",
        icon: Terminal,
    },
    {
        id: "design-decisions",
        label: "Design Decisions",
        icon: FileText,
    },
]

/* ─────────── copy button ─────────── */
function CopyButton({ text }: { text: string }) {
    const [copied, setCopied] = useState(false)

    return (
        <button
            onClick={() => {
                navigator.clipboard.writeText(text)
                setCopied(true)
                setTimeout(() => setCopied(false), 2000)
            }}
            className="absolute right-3 top-3 rounded-md border border-white/10 bg-white/5 p-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            title="Copy"
        >
            {copied ? <Check className="h-3.5 w-3.5 text-green-400" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
    )
}

/* ─────────── code block ─────────── */
function CodeBlock({ code, language = "python" }: { code: string; language?: string }) {
    return (
        <div className="relative group my-4 rounded-lg border border-border bg-[#0d1117] overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2 border-b border-white/5">
                <span className="text-xs font-mono text-muted-foreground">{language}</span>
            </div>
            <div className="relative">
                <CopyButton text={code} />
                <pre className="overflow-x-auto p-4 text-sm leading-relaxed">
                    <code className="font-mono text-green-300/90">{code}</code>
                </pre>
            </div>
        </div>
    )
}

/* ─────────── info card ─────────── */
function InfoCard({
    icon: Icon,
    title,
    children,
    variant = "default",
}: {
    icon: React.ElementType
    title: string
    children: React.ReactNode
    variant?: "default" | "green" | "amber" | "blue"
}) {
    const colors = {
        default: "border-border bg-card/50",
        green: "border-green-500/20 bg-green-500/5",
        amber: "border-amber-500/20 bg-amber-500/5",
        blue: "border-blue-500/20 bg-blue-500/5",
    }
    const iconColors = {
        default: "text-muted-foreground",
        green: "text-green-400",
        amber: "text-amber-400",
        blue: "text-blue-400",
    }

    return (
        <div className={`rounded-xl border p-5 ${colors[variant]} transition-all duration-200 hover:shadow-lg hover:shadow-black/5`}>
            <div className="flex items-center gap-3 mb-3">
                <div className={`flex h-8 w-8 items-center justify-center rounded-lg bg-background/50 ${iconColors[variant]}`}>
                    <Icon className="h-4 w-4" />
                </div>
                <h4 className="font-semibold text-foreground">{title}</h4>
            </div>
            <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
        </div>
    )
}

/* ─────────── section component ─────────── */
function Section({
    id,
    title,
    icon: Icon,
    children,
}: {
    id: string
    title: string
    icon: React.ElementType
    children: React.ReactNode
}) {
    return (
        <section id={id} className="scroll-mt-24 mb-16">
            <div className="flex items-center gap-3 mb-6">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-green-500/10 text-green-400 shadow-[0_0_15px_-5px_rgba(34,197,94,0.4)]">
                    <Icon className="h-5 w-5" />
                </div>
                <h2 className="text-2xl font-bold tracking-tight text-foreground">{title}</h2>
            </div>
            {children}
        </section>
    )
}

/* ─────────── sub-section component ─────────── */
function SubSection({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
    return (
        <div id={id} className="scroll-mt-24 mb-10">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <ArrowRight className="h-4 w-4 text-green-400" />
                {title}
            </h3>
            {children}
        </div>
    )
}

/* ─────────── data table ─────────── */
function DataTable({ headers, rows }: { headers: string[]; rows: string[][] }) {
    return (
        <div className="overflow-x-auto my-4 rounded-lg border border-border">
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-border bg-muted/30">
                        {headers.map((h) => (
                            <th key={h} className="px-4 py-3 text-left font-medium text-foreground">
                                {h}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, i) => (
                        <tr key={i} className="border-b border-border/50 hover:bg-muted/10 transition-colors">
                            {row.map((cell, j) => (
                                <td key={j} className="px-4 py-3 text-muted-foreground font-mono text-xs">
                                    {cell}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}

/* ─────────── sidebar TOC ─────────── */
function TableOfContents({ activeId }: { activeId: string }) {
    const [expanded, setExpanded] = useState<Record<string, boolean>>(
        Object.fromEntries(toc.map((item) => [item.id, true]))
    )

    return (
        <nav className="space-y-1">
            {toc.map((item) => {
                const Icon = item.icon
                const isActive = activeId === item.id || item.children?.some((c) => c.id === activeId)

                return (
                    <div key={item.id}>
                        <a
                            href={`#${item.id}`}
                            onClick={(e) => {
                                if (item.children) {
                                    e.preventDefault()
                                    setExpanded((prev) => ({ ...prev, [item.id]: !prev[item.id] }))
                                    const el = document.getElementById(item.id)
                                    if (el) el.scrollIntoView({ behavior: "smooth" })
                                }
                            }}
                            className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-all duration-200 ${isActive
                                ? "bg-green-500/10 text-green-400 font-medium"
                                : "text-muted-foreground hover:bg-muted hover:text-foreground"
                                }`}
                        >
                            <Icon className="h-4 w-4 shrink-0" />
                            <span className="flex-1 truncate">{item.label}</span>
                            {item.children &&
                                (expanded[item.id] ? (
                                    <ChevronDown className="h-3.5 w-3.5 shrink-0" />
                                ) : (
                                    <ChevronRight className="h-3.5 w-3.5 shrink-0" />
                                ))}
                        </a>

                        {item.children && expanded[item.id] && (
                            <div className="ml-6 mt-0.5 space-y-0.5 border-l border-border/50 pl-3">
                                {item.children.map((child) => (
                                    <a
                                        key={child.id}
                                        href={`#${child.id}`}
                                        className={`block rounded-md px-2.5 py-1.5 text-xs transition-colors ${activeId === child.id
                                            ? "text-green-400 font-medium"
                                            : "text-muted-foreground hover:text-foreground"
                                            }`}
                                    >
                                        {child.label}
                                    </a>
                                ))}
                            </div>
                        )}
                    </div>
                )
            })}
        </nav>
    )
}

/* ═══════════════════════════════════════════════════════════════════
   MAIN DOCUMENTATION PAGE
   ═══════════════════════════════════════════════════════════════════ */

export default function DocsPage() {
    const [activeId, setActiveId] = useState("overview")

    // Intersection observer for active section highlighting
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setActiveId(entry.target.id)
                    }
                })
            },
            { rootMargin: "-80px 0px -60% 0px", threshold: 0 }
        )

        const ids = toc.flatMap((item) => [item.id, ...(item.children?.map((c) => c.id) || [])])
        ids.forEach((id) => {
            const el = document.getElementById(id)
            if (el) observer.observe(el)
        })

        return () => observer.disconnect()
    }, [])

    return (
        <div className="flex gap-8 max-w-[1400px] mx-auto px-4 md:px-6 py-8">
            {/* ─── Sidebar TOC ─── */}
            <aside className="hidden lg:block w-64 shrink-0">
                <div className="sticky top-24 max-h-[calc(100vh-120px)] overflow-y-auto rounded-xl border border-border bg-card/30 backdrop-blur-sm p-4">
                    <div className="flex items-center gap-2 mb-4 pb-3 border-b border-border">
                        <FileText className="h-4 w-4 text-green-400" />
                        <span className="text-sm font-semibold text-foreground">Documentation</span>
                    </div>
                    <TableOfContents activeId={activeId} />
                </div>
            </aside>

            {/* ─── Main Content ─── */}
            <main className="flex-1 min-w-0">
                {/* Hero */}
                <div className="mb-12 pb-8 border-b border-border">
                    <div className="flex items-center gap-2 text-sm text-green-400 font-medium mb-3">
                        <Code2 className="h-4 w-4" />
                        <span>Architecture & Engineering</span>
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-foreground mb-4">
                        Stock Radar
                        <span className="bg-gradient-to-r from-green-400 to-emerald-300 bg-clip-text text-transparent">
                            {" "}Documentation
                        </span>
                    </h1>
                    <p className="text-lg text-muted-foreground max-w-2xl leading-relaxed">
                        A complete guide to every component, data flow, engineering pattern,
                        and design decision that powers the AI-driven stock analysis platform.
                    </p>
                    <div className="flex flex-wrap gap-3 mt-6">
                        {[
                            { label: "Python Backend", icon: Terminal },
                            { label: "Next.js Frontend", icon: Globe },
                            { label: "Multi-LLM AI", icon: Brain },
                            { label: "ML Predictions", icon: TrendingUp },
                            { label: "Vector Search", icon: Search },
                            { label: "Real-time Data", icon: Activity },
                        ].map(({ label, icon: I }) => (
                            <span
                                key={label}
                                className="inline-flex items-center gap-1.5 rounded-full bg-muted/50 border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground"
                            >
                                <I className="h-3 w-3" />
                                {label}
                            </span>
                        ))}
                    </div>
                </div>

                {/* ═══ 1. OVERVIEW ═══ */}
                <Section id="overview" title="What Is Stock Radar?" icon={Zap}>
                    <p className="text-muted-foreground leading-relaxed mb-6">
                        Stock Radar is an <strong className="text-foreground">AI-powered stock analysis platform</strong> that
                        brings institutional-grade intelligence to individual traders. Think of it as having a team of expert
                        analysts working 24/7 — one fetching market data, one crunching numbers using technical indicators,
                        one reading news and sentiment, one running machine learning models, and one sending you alerts when
                        something important happens.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                        <InfoCard icon={Database} title="Parallel Data Fetching" variant="green">
                            Fetches quotes, history, fundamentals, news, and social sentiment from 5+ sources simultaneously
                            using ThreadPoolExecutor — cutting latency from ~15s to ~3s.
                        </InfoCard>
                        <InfoCard icon={Brain} title="Multi-Model AI Analysis" variant="blue">
                            Uses a fallback chain of LLMs (Groq → Z.AI GLM → Gemini) with task-based model routing.
                            If one provider fails, the next one picks up automatically.
                        </InfoCard>
                        <InfoCard icon={BarChart3} title="Algorithmic Scoring" variant="amber">
                            Pure math-based scoring for Momentum, Value, Quality, and Risk — deterministic,
                            reproducible, and auditable. No AI needed.
                        </InfoCard>
                        <InfoCard icon={Shield} title="Production-Grade Safety" variant="green">
                            Kill switches, canary rollouts, guardrails, paper trading gates, and an immutable
                            audit trail. Enterprise-grade risk management.
                        </InfoCard>
                    </div>

                    <p className="text-sm text-muted-foreground italic">
                        Every component follows the <strong className="text-foreground">Single Responsibility Principle</strong> — each
                        class does exactly one thing, and does it well. If Yahoo Finance goes down, only the fetcher knows about it;
                        the analyzer, storage, and alerts keep working with whatever data is available.
                    </p>
                </Section>

                {/* ═══ 2. ARCHITECTURE ═══ */}
                <Section id="architecture" title="System Architecture" icon={Layers}>
                    <SubSection id="architecture-diagram" title="Architecture Diagram">
                        <CodeBlock
                            language="text"
                            code={`┌──────────────────────────────────────────────────────────────────────┐
│                               REQUESTS                               │
│  Browser / Dashboard                CLI                Slack/Telegram │
└──────────────┬───────────────────────┬───────────────────────────┬────┘
               │                       │                           │
               ▼                       ▼                           │
      Next.js /api/* routes         main.py                        │
               │                       │                           │
               ▼                       │                           │
      FastAPI backend (/v1/*)          │                           │
               │                       │                           │
               └──────────────┬────────┴───────────────────────────┘
                              ▼
                  StockRadar / ChatAssistant runtime
                              │
      ┌───────────────────────┼───────────────────────────┐
      │                       │                           │
      ▼                       ▼                           ▼
 StockFetcher           RAG / Retrieval              Notifications
 • Twelve Data          • Cohere embeddings          • Slack
 • Yahoo Finance        • Supabase RPC search        • Telegram
 • Finnhub              • pgvector cosine search
 • Reddit/ApeWisdom     • Exact DB lookups
      │                       │
      └──────────────┬────────┘
                     ▼
              StockAnalyzer / Chat LLM
              • Groq Llama 3.1 70B / 8B
              • Z.AI GLM-4.7
              • Gemini 2.5 Flash
                     │
                     ▼
                   Storage
              • Supabase/Postgres
              • analysis/news/chat rows
              • stored embeddings for future RAG
              • pgvector indexes + SQL RPC functions
┌──────────────────────────────────────────────────────────────────────┐
│ The LLM writes the answer/analysis. The embedding model only creates │
│ vectors so past records can be retrieved by meaning later.           │
└──────────────────────────────────────────────────────────────────────┘`}
                        />
                    </SubSection>

                    <SubSection id="design-philosophy" title="Design Philosophy">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <InfoCard icon={LayoutGrid} title="Single Responsibility" variant="default">
                                Each agent handles exactly one concern. The fetcher doesn&apos;t analyze; the analyzer
                                doesn&apos;t store; the storage doesn&apos;t notify. This makes each part testable and replaceable.
                            </InfoCard>
                            <InfoCard icon={Shield} title="Graceful Degradation" variant="default">
                                If any external dependency fails, the system continues with reduced functionality. If Finnhub is
                                down, the system uses Yahoo Finance. If the ML model isn&apos;t trained, algorithmic scoring takes over.
                            </InfoCard>
                            <InfoCard icon={Layers} title="Layered Architecture" variant="default">
                                Users → API → Orchestrator → Agents → External Services. Every layer only talks to its
                                immediate neighbor, never bypassing the hierarchy.
                            </InfoCard>
                        </div>
                    </SubSection>

                    <SubSection id="runtime-flow" title="Runtime Request Flows">
                        <p className="text-sm text-muted-foreground mb-4">
                            <strong className="text-foreground">Runtime</strong> means the code path that executes while a live
                            request is being handled. In this codebase, that means a browser hit to a Next.js route, a backend
                            job running in FastAPI, or a CLI command running <code className="text-green-400 text-xs">main.py</code>.
                            It does <em>not</em> mean every config field or helper method that exists in the repo.
                        </p>
                        <DataTable
                            headers={["Flow", "Entry Point", "Core Path", "What Actually Happens"]}
                            rows={[
                                ["Analyze stock", "/api/analyze", "Next.js → FastAPI job → StockRadar.analyze_stock()", "Fetch market data, optionally retrieve RAG context, run LLM analysis, store result + embedding, create signal + alert"],
                                ["Chat ask", "/api/ask", "Next.js → FastAPI /v1/ask → StockChatAssistant.ask()", "Exact symbol lookups + semantic retrieval, then LLM answer generation, then chat history storage"],
                                ["CLI analyze", "python main.py analyze ...", "CLI → StockRadar.analyze_stock()", "Same backend analysis pipeline without the web proxy layer"],
                            ]}
                        />
                        <CodeBlock
                            language="text"
                            code={`ANALYZE REQUEST
Browser → /api/analyze → backend /v1/analyze/jobs → background job
→ StockRadar.analyze_stock()
→ StockFetcher gets quote/history/fundamentals/news/social
→ RAG retrieval tries semantic lookup of similar past records
→ LLM writes the actual analysis JSON
→ result stored in Supabase
→ Cohere embedding generated for the stored analysis text
→ future requests can retrieve that record semantically

CHAT REQUEST
Browser → /api/ask → backend /v1/ask → StockChatAssistant.ask()
→ ensure data exists for detected symbols
→ semantic retrieval of similar analyses/news/knowledge/chat
→ exact lookups for latest analysis/news/indicators
→ LLM writes the answer
→ chat messages stored with embeddings for future retrieval`}
                        />
                    </SubSection>
                </Section>

                {/* ═══ 3. ORCHESTRATOR ═══ */}
                <Section id="orchestrator" title="The Orchestrator — StockRadar" icon={Workflow}>
                    <p className="text-muted-foreground leading-relaxed mb-6">
                        <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">main.py</code> contains
                        the <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">StockRadar</code> class —
                        the conductor of the entire system. Like a restaurant manager, it doesn&apos;t cook food (that&apos;s StockFetcher),
                        design menus (StockAnalyzer), take payments (StockStorage), or seat guests (NotificationManager).
                        It <strong className="text-foreground">coordinates</strong> all of them.
                    </p>

                    <SubSection id="orchestrator-init" title="Initialization">
                        <CodeBlock code={`class StockRadar:
    def __init__(self):
        # 1. Start Prometheus metrics server
        self._metrics_server = start_metrics_server(port=9090)

        # 2. Create each specialist agent
        self.fetcher = StockFetcher()           # Data collector
        self.analyzer = StockAnalyzer()         # AI brain
        self.storage = StockStorage()           # Database layer
        self.notifications = NotificationManager()  # Alert system

        # 3. Start real-time WebSocket feed
        self._realtime = get_realtime_manager()
        self._realtime.start()

        # 4. Verify database schema
        self.storage.ensure_schema()`} />
                        <p className="text-sm text-muted-foreground">
                            Each agent is instantiated independently. If Finnhub is down, the realtime feed won&apos;t start —
                            but the rest of the system works perfectly with historical data. This is <strong className="text-foreground">graceful degradation</strong> in action.
                        </p>
                    </SubSection>

                    <SubSection id="analysis-pipeline" title="5-Step Analysis Pipeline">
                        <div className="space-y-3">
                            {[
                                {
                                    step: "1",
                                    title: "FETCH",
                                    desc: "Mode-aware history resolution (intraday → 5d/15m bars, longterm → 5y/weekly bars). Up to 6 API calls in parallel via ThreadPoolExecutor: quote, history, fundamentals, Yahoo news, Finnhub news, Finnhub sentiment. Checks WebSocket cache for instant quotes. Then calculates RSI, MACD, Bollinger Bands, ATR, VWAP, ADX. Also fetches Reddit social sentiment.",
                                    color: "text-blue-400",
                                },
                                {
                                    step: "2",
                                    title: "ANALYZE",
                                    desc: "Retrieves RAG context using semantic search over stored embeddings plus exact DB lookups where needed. Builds a structured prompt — intraday includes social sentiment & real-time buzz; longterm includes 50+ fundamentals (valuation, profitability, growth, health, dividends, analyst consensus). Calls the LLM fallback chain to write the actual analysis. RAG Validation grades the output (A-F) on faithfulness, relevancy, groundedness & temporal validity.",
                                    color: "text-purple-400",
                                },
                                {
                                    step: "3",
                                    title: "VERIFY + ALGO",
                                    desc: "A second LLM cross-checks the analysis. Then algo prediction: tries RegimeAwarePredictor → SignalPredictor (37+ features incl. Phase-5 factors, Phase-6 FinBERT sentiment, Finnhub sentiment). Falls back to StockScorer formulas. Classifies market regime, sizes positions, computes ATR-based stop-loss/take-profit, auto-scales via per-trade risk budgeting (2% max). Returns full score breakdowns.",
                                    color: "text-amber-400",
                                },
                                {
                                    step: "4",
                                    title: "STORE",
                                    desc: "Saves to Supabase: stock record, price data, indicators, analysis row, and a Cohere embedding of the analysis text for future semantic retrieval. Signals are stored separately for alerting and history.",
                                    color: "text-green-400",
                                },
                                {
                                    step: "5",
                                    title: "ALERT",
                                    desc: "If the signal is actionable (buy/sell), sends rich notifications to Slack (block messages) and Telegram (HTML). Records delivery status. Sends API usage summary.",
                                    color: "text-rose-400",
                                },
                            ].map(({ step, title, desc, color }) => (
                                <div key={step} className="flex gap-4 items-start p-4 rounded-lg border border-border bg-card/30 hover:bg-card/50 transition-colors">
                                    <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-background font-bold text-sm ${color}`}>
                                        {step}
                                    </div>
                                    <div>
                                        <span className={`font-semibold text-sm ${color}`}>{title}</span>
                                        <p className="text-sm text-muted-foreground mt-1">{desc}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </SubSection>
                </Section>

                {/* ═══ 4. AGENTS ═══ */}
                <Section id="agents" title="Agent Layer — The Specialist Workers" icon={Cpu}>
                    {/* 4.1 Fetcher */}
                    <SubSection id="fetcher" title="StockFetcher — The Data Collector">
                        <p className="text-muted-foreground text-sm mb-4">
                            <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">src/agents/fetcher.py</code> (1,221 lines)
                            — The researcher who gathers all raw information and brings it back organized.
                        </p>
                        <DataTable
                            headers={["Data Type", "Primary Source", "Fallback", "Method"]}
                            rows={[
                                ["Live Quotes", "Twelve Data", "Yahoo Finance", "REST API"],
                                ["Price History", "Twelve Data", "Yahoo Finance", "REST API"],
                                ["Fundamentals (50+)", "Yahoo Finance", "—", "REST API"],
                                ["News", "Yahoo + Finnhub", "Yahoo alone", "REST API"],
                                ["Aggregate Sentiment", "Finnhub", "—", "REST API"],
                                ["Reddit Buzz", "ApeWisdom", "—", "REST API (free)"],
                            ]}
                        />
                        <h4 className="font-medium text-foreground text-sm mt-6 mb-2">Mode-Aware History Resolution</h4>
                        <p className="text-sm text-muted-foreground mb-2">
                            The fetcher tailors the history window and bar size to the analysis mode via
                            <code className="text-green-400 bg-green-500/10 px-1 rounded text-xs"> _resolve_analysis_history_config(mode, period)</code>:
                        </p>
                        <DataTable
                            headers={["Mode", "Period", "Window", "Bar Size", "Why"]}
                            rows={[
                                ["Intraday", "1d", "1 day", "5-min bars", "Micro price action for day trading"],
                                ["Intraday", "5d (default)", "5 days", "15-min bars", "Multi-day patterns without noise"],
                                ["Longterm", "5y (default)", "5 years", "Weekly bars", "Secular trends and cycles"],
                                ["Longterm", "max", "Max available", "Weekly bars", "Full company history"],
                            ]}
                        />
                        <p className="text-sm text-muted-foreground mt-4 mb-2">
                            <strong className="text-foreground">The Parallel Fetching Pattern:</strong> Up to 6 API calls run simultaneously.
                            WebSocket cache skips the quote call when fresh data ({'<'}60s) is available:
                        </p>
                        <CodeBlock code={`with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {}
    if not rt_quote:  # Skip if WebSocket cache has fresh data
        futures["quote"] = pool.submit(self.get_quote, symbol)
    futures["history"] = pool.submit(
        self.get_price_history, symbol, period, interval)
    futures["fundamentals"] = pool.submit(self.get_fundamentals, symbol)
    futures["news_yahoo"] = pool.submit(self.get_news_yahoo, symbol)
    if self.finnhub_key:
        futures["news_finnhub"] = pool.submit(
            self.get_news_finnhub, symbol)
        futures["finnhub_sentiment"] = pool.submit(
            self.get_sentiment_finnhub, symbol)
    # All calls run simultaneously → ~3s instead of 18s`} />
                        <h4 className="font-medium text-foreground text-sm mt-6 mb-2">Enhanced Fundamentals (50+ Data Points)</h4>
                        <DataTable
                            headers={["Category", "Metrics"]}
                            rows={[
                                ["Company Info", "Name, sector, industry, website, HQ, employees, description"],
                                ["Valuation", "P/E, Forward P/E, PEG, P/B, P/S"],
                                ["EPS & Revenue", "EPS TTM, Forward EPS, Revenue, Gross Profit, EBITDA, Net Income"],
                                ["Shares & Float", "Outstanding, Float, Short Interest, Short Ratio, Insider/Institutional %"],
                                ["Risk & Volatility", "Beta, 52-Week High/Low, 50-Day Avg, 200-Day Avg"],
                                ["Profitability", "Profit/Operating/Gross Margin, ROE, ROA"],
                                ["Growth", "Revenue Growth, Earnings Growth, Quarterly Growth"],
                                ["Dividends", "Yield, Rate, Payout Ratio, Ex-Dividend Date"],
                                ["Financial Health", "Current Ratio, Quick Ratio, D/E, Cash, Debt, FCF, Operating CF"],
                                ["Analyst Data", "Target High/Low/Mean/Median, Recommendation, Analyst Count"],
                            ]}
                        />
                        <p className="text-sm text-muted-foreground mt-3">
                            <strong className="text-foreground">Technical Indicators</strong> are calculated from raw price data
                            using standard financial math — no external libraries. RSI (overbought/oversold), MACD (momentum),
                            Bollinger Bands (volatility), ATR (risk sizing & position scaling), VWAP (institutional benchmark), ADX (trend strength).
                        </p>
                    </SubSection>

                    {/* 4.2 Analyzer */}
                    <SubSection id="analyzer" title="StockAnalyzer — The AI Brain">
                        <p className="text-muted-foreground text-sm mb-4">
                            <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">src/agents/analyzer.py</code> (1,202 lines)
                            — The senior analyst who reads all data, applies expertise, and delivers a recommendation.
                        </p>
                        <h4 className="font-medium text-foreground text-sm mb-2">Inference Models vs Embedding Model</h4>
                        <DataTable
                            headers={["Role", "Used By", "Default Runtime Model(s)", "What It Does"]}
                            rows={[
                                ["Inference model", "StockAnalyzer (analysis/algo)", "Groq Llama 3.1 70B → Z.AI GLM-4.7 → Gemini 2.5 Flash", "Writes the stock analysis, reasoning, targets, and structured JSON output"],
                                ["Inference model", "ChatAssistant (chat)", "Groq Llama 3.1 70B → Z.AI GLM-4.7 → Gemini 2.5 Flash", "Writes the conversational answer after context is prepared"],
                                ["Inference model", "News / sentiment tasks", "Groq Llama 3.1 8B Instant → Gemini 2.5 Flash → Z.AI GLM-4.7", "Handles lighter summarization/classification-style tasks"],
                                ["Embedding model", "StockStorage / RAG", "Cohere embed-english-v3.0", "Turns text into vectors for semantic retrieval; it does not write the final analysis"],
                                ["Embedding fallback", "Storage / search", "None if COHERE_API_KEY missing", "Semantic retrieval becomes empty, but exact lookups and LLM analysis still continue"],
                            ]}
                        />
                        <p className="text-sm text-muted-foreground mt-2 mb-6">
                            The key split is simple: <strong className="text-foreground">LLMs generate answers</strong>, while
                            <strong className="text-foreground"> embeddings generate vectors</strong>. Vectors are used to find
                            related records by meaning before or after the LLM call.
                        </p>
                        <h4 className="font-medium text-foreground text-sm mb-2">Task-Based Model Routing</h4>
                        <DataTable
                            headers={["Task", "Model 1 (Primary)", "Model 2", "Model 3"]}
                            rows={[
                                ["analysis", "Groq Llama 70B", "Z.AI GLM-4.7", "Gemini Flash"],
                                ["algo", "Groq Llama 70B", "Z.AI GLM-4.7", "Gemini Flash"],
                                ["news", "Groq Llama 8B", "Gemini Flash", "Z.AI GLM-4.7"],
                                ["sentiment", "Groq Llama 8B", "Gemini Flash", "Z.AI GLM-4.7"],
                                ["chat", "Groq Llama 70B", "Z.AI GLM-4.7", "Gemini Flash"],
                            ]}
                        />
                        <p className="text-sm text-muted-foreground mt-2 mb-6">
                            <strong className="text-foreground">Why different models?</strong> News summarization doesn&apos;t need a 70B model — a faster 8B
                            gives the same quality at 5× the speed. Stock analysis needs deeper reasoning, so we use the larger model.
                        </p>

                        <h4 className="font-medium text-foreground text-sm mb-2">Enhanced Intraday Analysis</h4>
                        <p className="text-sm text-muted-foreground mb-2">
                            The intraday prompt weaves together six distinct data layers:
                        </p>
                        <DataTable
                            headers={["Data Layer", "What It Tells the LLM", "Source"]}
                            rows={[
                                ["Current Price", "Where the stock is right now (price, change%, volume vs avg)", "get_quote()"],
                                ["Technical Indicators", "Bullish/bearish technicals (RSI, MACD, SMA, Bollinger)", "calculate_indicators()"],
                                ["Recent News", "External catalysts — earnings, partnerships, macro", "get_news_yahoo() + get_news_finnhub()"],
                                ["Social Sentiment", "Reddit mentions, rank, overall community buzz", "get_social_sentiment()"],
                                ["RAG Context", "How have similar setups played out historically?", "RAGRetriever.retrieve_context()"],
                                ["RAG Validation", "Post-analysis grading: faithfulness, groundedness (A-F)", "RAGValidator.validate_analysis()"],
                            ]}
                        />

                        <h4 className="font-medium text-foreground text-sm mt-6 mb-2">Enhanced Long-term Analysis</h4>
                        <p className="text-sm text-muted-foreground mb-2">
                            The longterm prompt is structured as a professional fundamental analysis with nine sections:
                        </p>
                        <DataTable
                            headers={["Section", "Data Included"]}
                            rows={[
                                ["1. Current Price", "Price, 52-Week High/Low, Market Cap"],
                                ["2. Valuation", "P/E, Forward P/E, PEG, P/B, P/S"],
                                ["3. Profitability", "Profit Margin, Operating Margin, ROE, ROA"],
                                ["4. Growth", "Revenue Growth, Earnings Growth"],
                                ["5. Financial Health", "Current Ratio, D/E, Free Cash Flow"],
                                ["6. Dividends", "Yield, Payout Ratio"],
                                ["7. Analyst Consensus", "Mean Target Price, Recommendation"],
                                ["8. Technicals (Weekly)", "RSI, Price vs SMA(50), SMA(200)"],
                                ["9. RAG Context", "Recent news + similar past analyses"],
                            ]}
                        />
                        <p className="text-sm text-muted-foreground mt-2">
                            Both modes run RAG Validation post-analysis. Longterm mode uses <code className="text-green-400 text-xs">analysis_mode=&quot;longterm&quot;</code> which
                            relaxes temporal freshness requirements — weekly data is acceptable for long-term context.
                        </p>
                    </SubSection>

                    {/* 4.3 Storage */}
                    <SubSection id="storage" title="StockStorage — The Memory">
                        <p className="text-muted-foreground text-sm mb-4">
                            <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">src/agents/storage.py</code> (1,724 lines)
                            — The filing cabinet that remembers everything, and can find related documents by meaning.
                        </p>
                        <DataTable
                            headers={["Table", "Purpose", "Vector-Enabled?"]}
                            rows={[
                                ["users", "User accounts & preferences", "No"],
                                ["stocks", "Master stock list", "No"],
                                ["watchlist", "User's tracked stocks", "No"],
                                ["price_history", "Historical OHLCV data", "No"],
                                ["analysis", "AI analysis results + embeddings", "✓ pgvector"],
                                ["signals", "Actionable trading signals + context embeddings", "✓ pgvector"],
                                ["alerts", "Notification delivery records", "No"],
                                ["news", "Stored news articles + embeddings", "✓ pgvector"],
                                ["chat_history", "Stored chat turns + embeddings", "✓ pgvector"],
                                ["knowledge_base", "RAG knowledge base", "✓ pgvector"],
                            ]}
                        />
                        <InfoCard icon={Search} title="Vector Embeddings — Semantic Search" variant="blue">
                            At runtime, analyses, signals, news, chat turns, and knowledge entries are embedded with the configured
                            provider. By default, the app uses Cohere
                            (<code className="text-green-400 text-xs">embed-english-v3.0</code>) and stores 1024-dimensional vectors
                            in Postgres/pgvector. Later, a query is embedded too, and Postgres/pgvector compares vector similarity so the app can find
                            records that are <em>meaningfully similar</em>, not just exact text matches.
                        </InfoCard>
                    </SubSection>

                    <SubSection id="search-types" title="Search Types Used In The App">
                        <p className="text-muted-foreground text-sm mb-4">
                            The app does <strong className="text-foreground">not</strong> use one single search strategy. It mixes
                            exact lookups, recency queries, filters, and semantic search depending on the job.
                        </p>
                        <DataTable
                            headers={["Search Type", "Example In Stock Radar", "Backed By", "Why It Exists"]}
                            rows={[
                                ["Exact lookup", "Find stock row by symbol", "SQL equality filter", "Reliable identity lookup; best when you know the exact key"],
                                ["Recency lookup", "Latest analysis / recent news", "ORDER BY created_at / published_at", "Gives the freshest known state"],
                                ["Filter search", "Mode=user=symbol=sentiment constraints", "SQL WHERE clauses", "Structured retrieval by known metadata"],
                                ["Semantic search", "Similar analyses / signals / news / knowledge by meaning", "Configured embedding provider + pgvector cosine similarity", "Finds related content even when wording differs"],
                                ["Hybrid search", "Not fully implemented in the core path yet", "Would combine exact/keyword + semantic", "Usually the most robust search pattern for production RAG systems"],
                            ]}
                        />
                        <InfoCard icon={Database} title="Semantic Search vs Exact Search" variant="amber">
                            Exact search answers questions like &quot;give me the row where symbol = AAPL&quot;.
                            Semantic search answers questions like &quot;give me records that mean something similar to
                            &apos;bullish breakout after earnings&apos;&quot;. Chat uses both: exact symbol lookups for the latest
                            stored facts, and semantic retrieval for historical context.
                        </InfoCard>
                    </SubSection>

                    {/* 4.4 Notifications */}
                    <SubSection id="notifications" title="NotificationManager — The Messenger">
                        <p className="text-muted-foreground text-sm mb-4">
                            <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">src/agents/alerts.py</code> (973 lines)
                            — Sends rich, color-coded alerts to Slack (block messages with emojis) and Telegram (HTML formatting).
                            Includes retry logic for rate-limited API calls.
                        </p>
                    </SubSection>

                    {/* 4.5 Scorer */}
                    <SubSection id="scorer" title="StockScorer — The Formula Engine">
                        <p className="text-muted-foreground text-sm mb-4">
                            <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">src/agents/scorer.py</code> (852 lines)
                            — Scores are <strong className="text-foreground">algorithmic, not AI-generated</strong>. Deterministic, reproducible, auditable.
                        </p>
                        <DataTable
                            headers={["Score", "Range", "Based On", "Example"]}
                            rows={[
                                ["Momentum", "0-100", "RSI, MACD, Price vs SMA", "RSI<30 = +30 pts (oversold)"],
                                ["Value", "0-100", "P/E, P/B, Dividend, P/S", "P/E<15 = +30 pts (undervalued)"],
                                ["Quality", "0-100", "ROE, Margins, Debt/Equity", "ROE>20% = excellent"],
                                ["Risk", "1-10", "ATR, ADX, Debt levels", "High ATR = high risk"],
                                ["Confidence", "0-100", "Data availability", "More data = higher confidence"],
                            ]}
                        />
                        <p className="text-sm text-muted-foreground mt-2">
                            Supports configurable weight presets: <code className="text-green-400 text-xs">balanced</code>,{" "}
                            <code className="text-green-400 text-xs">momentum_focus</code>,{" "}
                            <code className="text-green-400 text-xs">value_focus</code>,{" "}
                            <code className="text-green-400 text-xs">conservative</code>.
                        </p>
                    </SubSection>

                    {/* 4.6 Chat */}
                    <SubSection id="chat" title="ChatAssistant — The Conversationalist">
                        <p className="text-muted-foreground text-sm mb-4">
                            <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">src/agents/chat_assistant.py</code> (1,052 lines)
                            — RAG-powered conversational assistant. Before answering, it performs two retrieval passes:
                            semantic search for similar historical context, and exact symbol lookups for the latest stored facts.
                        </p>
                        <CodeBlock
                            language="text"
                            code={`1. Detect symbols from the question
2. Ensure local data exists for those symbols
3. Run semantic retrieval across analyses/news/knowledge/chat
4. Run exact lookups for latest analysis, indicators, and recent news
5. Build one prompt containing both direct facts and RAG context
6. Call chat LLM route
7. Store user + assistant chat turns with embeddings for future retrieval`}
                        />
                    </SubSection>

                    {/* 4.7 RAG */}
                    <SubSection id="rag" title="RAG System — Retriever & Validator">
                        <p className="text-muted-foreground text-sm mb-4">
                            <strong className="text-foreground">RAGRetriever</strong>{" "}
                            <code className="text-green-400 bg-green-500/10 px-1 rounded text-xs">rag_retriever.py</code> — Searches
                            similar past analyses, signals, related news, knowledge entries, and chat history by vector similarity.
                        </p>
                        <p className="text-muted-foreground text-sm mb-4">
                            <strong className="text-foreground">RAGValidator</strong>{" "}
                            <code className="text-green-400 bg-green-500/10 px-1 rounded text-xs">rag_validator.py</code> — RAGAS-style
                            metrics: Faithfulness, Context Relevancy, Groundedness, Temporal Validity. Each analysis gets a letter grade (A-F).
                        </p>
                        <InfoCard icon={Workflow} title="Current Analysis-Time Retrieval" variant="blue">
                            The analyzer now builds its retrieval query from the live market state, including RSI, MACD, price,
                            and change%. That gives the RAG lookup a more specific semantic query than a generic
                            <code className="text-green-400 text-xs"> AAPL intraday analysis</code> string and makes past-context retrieval more relevant.
                        </InfoCard>
                    </SubSection>

                    {/* 4.8 Realtime */}
                    <SubSection id="realtime" title="RealtimeManager — The Live Wire">
                        <p className="text-muted-foreground text-sm mb-4">
                            <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">src/agents/realtime.py</code> (254 lines)
                            — Finnhub WebSocket for real-time trade data (&lt;100ms latency). Runs in a background daemon thread.
                            Caches latest prices in-memory so <code className="text-green-400 text-xs">get_quote()</code> returns instantly.
                        </p>
                    </SubSection>

                    {/* 4.9 Usage Tracker */}
                    <SubSection id="usage-tracker" title="UsageTracker — The Accountant">
                        <p className="text-muted-foreground text-sm">
                            <code className="text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded text-xs">src/agents/usage_tracker.py</code> (349 lines)
                            — Tracks API usage across all providers with threshold alerts at 50%, 75%, 90%, 95%, 100%
                            and auto-reset based on daily/monthly periods.
                        </p>
                    </SubSection>
                </Section>

                {/* ═══ 5. ML ═══ */}
                <Section id="ml" title="ML & Training Subsystem" icon={Brain}>
                    <SubSection id="predictor" title="SignalPredictor — The ML Model">
                        <p className="text-muted-foreground text-sm mb-2">
                            Loads a trained scikit-learn model (<code className="text-green-400 text-xs">.joblib</code>)
                            and predicts signals from a 37+ feature vector including technical indicators,
                            Phase-5 factor/microstructure features, and Phase-6 FinBERT sentiment + Finnhub aggregate sentiment.
                        </p>
                        <p className="text-sm text-muted-foreground mb-4">
                            <strong className="text-foreground">Regime-Aware Routing:</strong> Tries <code className="text-green-400 text-xs">RegimeAwarePredictor</code> first
                            (discovers bull/bear/volatile regime-specific models), then falls back to the general <code className="text-green-400 text-xs">SignalPredictor</code>.
                            The predictor auto-detects feature count from model metadata — models trained with 20 or 37 features both work transparently.
                        </p>
                    </SubSection>

                    <SubSection id="paper-trading" title="Paper Trading — The Simulator">
                        <p className="text-muted-foreground text-sm mb-4">
                            Complete simulated trading environment with position tracking, stop-loss/take-profit hits,
                            and rolling performance metrics (Sharpe ratio, max drawdown, win rate).
                        </p>
                        <h4 className="font-medium text-foreground text-sm mb-2">Promotion Gates (Must Pass ALL)</h4>
                        <DataTable
                            headers={["Gate", "Requirement"]}
                            rows={[
                                ["Min Trades", "≥ 10 completed"],
                                ["Sharpe Ratio", "> 0.0"],
                                ["Max Drawdown", "< -20%"],
                                ["Win Rate", "> 40%"],
                                ["Turnover", "< 5.0"],
                            ]}
                        />
                    </SubSection>

                    <SubSection id="broker" title="Broker Adapter — The Execution Layer">
                        <p className="text-muted-foreground text-sm mb-4">
                            Abstract interface for order execution with <strong className="text-foreground">idempotent orders</strong> (same
                            order_id = cached fill, no double-execution) and <strong className="text-foreground">exponential backoff retry</strong>.
                            Currently implements PaperBroker; designed for adding live brokers (Alpaca, IBKR).
                        </p>
                    </SubSection>

                    <SubSection id="risk-mgmt" title="Risk Management — The Safety Net">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <InfoCard icon={Lock} title="Kill Switches" variant="amber">
                                Max daily loss → HALT. Stale data → HALT. Excessive slippage → HALT.
                                Automatic safety mechanisms that stop all trading.
                            </InfoCard>
                            <InfoCard icon={Shield} title="Pre-Trade Risk" variant="amber">
                                Max position size, sector concentration limits, daily loss limits,
                                total exposure caps. Checked before every trade.
                            </InfoCard>
                            <InfoCard icon={Activity} title="Canary Rollout" variant="green">
                                Trade only approved symbols first. Monitor P&amp;L on canary set.
                                Auto-disable if canary shows losses. Google/Facebook-style deployment.
                            </InfoCard>
                            <InfoCard icon={FileText} title="Audit Trail" variant="blue">
                                Every signal, order, fill, risk check, and kill switch activation
                                recorded in an immutable append-only log. Daily summary reports.
                            </InfoCard>
                        </div>
                    </SubSection>
                </Section>

                {/* ═══ 6. INFRASTRUCTURE ═══ */}
                <Section id="infra" title="Infrastructure & Cross-Cutting" icon={Settings}>
                    <SubSection id="config" title="Configuration Management">
                        <p className="text-muted-foreground text-sm mb-2">
                            Pydantic BaseSettings for type-safe config. Priority: Env vars → .env file → Defaults.
                        </p>
                    </SubSection>

                    <SubSection id="config-reality" title="Current Runtime Reality">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <InfoCard icon={Brain} title="What Is Wired Through Settings" variant="green">
                                LLM routing is genuinely controlled through settings and env vars:
                                fallback order, task routes, API keys, guardrails, and most runtime toggles.
                            </InfoCard>
                            <InfoCard icon={Search} title="What Still Needs Care" variant="amber">
                                Embedding provider, model, and dimension are now runtime-configurable through settings and env vars.
                                The remaining hard boundary is the database schema:
                                vector columns and RPC functions are still sized for
                                <code className="text-green-400 text-xs"> 1024</code>-dimensional vectors unless you run a wider-schema migration.
                            </InfoCard>
                        </div>
                        <p className="text-sm text-muted-foreground mt-4">
                            In practice, the runtime is now provider-aware for embeddings as well as LLMs.
                            The main remaining operational constraint is keeping the configured embedding dimension aligned
                            with the pgvector schema used in Supabase.
                        </p>
                    </SubSection>

                    <SubSection id="caching" title="Caching Strategy">
                        <p className="text-sm text-muted-foreground">
                            The standalone <code className="text-green-400 text-xs">cache.py</code> module was removed.
                            Caching is now handled at the application level — Supabase handles data persistence,
                            the frontend uses React state for quote snapshots, and the usage tracker manages its own counters.
                            Cache-related Prometheus counters (<code className="text-green-400 text-xs">CACHE_HITS</code>, <code className="text-green-400 text-xs">CACHE_MISSES</code>) still exist in <code className="text-green-400 text-xs">metrics.py</code> for future use.
                        </p>
                    </SubSection>

                    <SubSection id="guardrails" title="LLM Output Guardrails">
                        <ol className="space-y-2 text-sm text-muted-foreground list-decimal list-inside">
                            <li><strong className="text-foreground">Schema check:</strong> Required JSON fields present?</li>
                            <li><strong className="text-foreground">Signal check:</strong> One of 5 valid values?</li>
                            <li><strong className="text-foreground">Confidence cap:</strong> Capped at 0.95 max (LLMs shouldn&apos;t be 100% sure about stocks)</li>
                            <li><strong className="text-foreground">Price check:</strong> Target within ±50% of current? (catches hallucinated prices)</li>
                            <li><strong className="text-foreground">Reasoning check:</strong> Explanation &gt;20 words?</li>
                            <li><strong className="text-foreground">Consistency:</strong> &quot;buy&quot; signal has bullish reasoning? (catches contradictions)</li>
                        </ol>
                    </SubSection>

                    <SubSection id="prompts" title="Prompt Versioning & A/B Testing">
                        <p className="text-muted-foreground text-sm">
                            Prompts are versioned templates (v1, v2) that can be A/B tested. Different prompt versions
                            produce different analysis styles — enabling systematic prompt engineering.
                        </p>
                    </SubSection>

                    <SubSection id="streaming" title="SSE Streaming">
                        <p className="text-muted-foreground text-sm">
                            Streams LLM tokens to the frontend in real-time using Server-Sent Events.
                            Users see tokens appear instantly instead of waiting 5-10 seconds.
                        </p>
                    </SubSection>

                    <SubSection id="metrics-infra" title="Prometheus Metrics">
                        <p className="text-sm text-muted-foreground mb-3">
                            Served natively at <code className="text-green-400 text-xs">GET /metrics</code> in the FastAPI backend (<code className="text-green-400 text-xs">backend/app.py:143</code>).
                            The standalone metrics server is disabled in the backend via <code className="text-green-400 text-xs">STOCK_RADAR_DISABLE_METRICS_SERVER=1</code> —
                            it only runs in CLI mode.
                        </p>
                        <DataTable
                            headers={["Metric", "What It Measures"]}
                            rows={[
                                ["llm_latency_seconds", "How long each LLM call takes"],
                                ["llm_requests_total", "Total LLM calls by model & status"],
                                ["llm_fallback_total", "How often fallback models trigger"],
                                ["llm_tokens_total", "Token consumption (input/output by model)"],
                                ["analysis_confidence", "Distribution of confidence scores"],
                                ["analysis_signal_total", "Count of each signal type by mode"],
                                ["analysis_duration_seconds", "Total pipeline duration by mode"],
                                ["api_cost_usd_total", "Running cost in USD by provider"],
                                ["data_fetch_latency_seconds", "Data provider fetch latency"],
                                ["scoring_composite", "Distribution of composite algo scores"],
                                ["guardrail_triggers_total", "How often guardrails fire"],
                                ["system_up", "Whether the system is running (lifecycle gauge)"],
                                ["ml_model_loaded", "Whether an ML model is loaded"],
                            ]}
                        />
                    </SubSection>

                    <SubSection id="token-accounting" title="Token Accounting & Cost">
                        <p className="text-muted-foreground text-sm">
                            Every API response includes a <code className="text-green-400 text-xs">meta</code> block with full
                            cost transparency — tokens in/out, cost in USD, model used, data sources, and latency.
                        </p>
                    </SubSection>
                </Section>

                {/* ═══ 7. API ═══ */}
                <Section id="api" title="Backend API (FastAPI)" icon={Server}>
                    <DataTable
                        headers={["Endpoint", "Method", "Description"]}
                        rows={[
                            ["/v1/analyze/jobs", "POST", "Start async analysis job (returns job ID)"],
                            ["/v1/analyze/jobs/{job_id}", "GET", "Check analysis job status"],
                            ["/v1/ask", "POST", "Chat assistant Q&A (with session history)"],
                            ["/v1/fundamentals", "GET", "Stock fundamentals"],
                            ["/v1/agent/momentum", "GET", "Momentum analysis with score breakdown"],
                            ["/v1/agent/rsi-divergence", "GET", "RSI divergence detection"],
                            ["/v1/agent/social-sentiment", "GET", "Reddit/social buzz"],
                            ["/v1/agent/support-resistance", "GET", "Pivot points + Bollinger + ATR"],
                            ["/v1/agent/stock-score", "GET", "Full algorithmic scores"],
                            ["/v1/agent/news-impact", "GET", "News sentiment with sector context"],
                            ["/metrics", "GET", "Prometheus metrics (FastAPI-native, no auth)"],
                            ["/health", "GET", "System health check with dependency status"],
                        ]}
                    />
                    <p className="text-sm text-muted-foreground mt-2">
                        Auth via <code className="text-green-400 text-xs">X-Backend-Api-Key</code> using <code className="text-green-400 text-xs">BACKEND_API_KEY</code> on all <code className="text-green-400 text-xs">/v1/</code> routes.
                        Long-running analyses execute in background thread pools (max_workers=2, 30-min TTL).
                        Analysis payloads now include a <code className="text-green-400 text-xs">guardrails</code> key with pass/fail status and issue details.
                    </p>
                </Section>

                {/* ═══ 8. PIPELINE WALKTHROUGH ═══ */}
                <Section id="pipeline-walkthrough" title="Full Pipeline Walkthrough" icon={GitBranch}>
                    <p className="text-muted-foreground text-sm mb-4">
                        Here are the two main runtime journeys: <strong className="text-foreground">analyze</strong> and
                        <strong className="text-foreground"> chat</strong>. These are the real request flows that execute while the app is live.
                    </p>
                    <SubSection id="analyze-walkthrough" title="Analyze Request Lifecycle">
                        <p className="text-sm text-muted-foreground mb-4">
                            The analyze path is intentionally <strong className="text-foreground">asynchronous</strong>. The browser does not sit on one long
                            request while market data is fetched, indicators are computed, RAG context is retrieved, the LLM runs, guardrails validate the
                            output, and Supabase writes complete. Instead, the web layer creates a background job and the UI polls for the final result.
                        </p>
                        <DataTable
                            headers={["Stage", "What Happens Technically", "Why It Exists"]}
                            rows={[
                                ["1. Browser -> Next.js route", "The dashboard sends POST /api/analyze with symbol, mode, and period. The route validates symbol shape, enforces allowed modes, normalizes the requested period, and applies route-level rate limiting before any backend call is made.", "This stops malformed or abusive requests early and keeps the browser-facing contract stable."],
                                ["2. Next.js -> backend job creation", "The route proxies the request to POST /v1/analyze/jobs using the backend API key header. FastAPI validates again and hands the work to the background job manager, which immediately returns HTTP 202 with a job id.", "Analysis is too heavy for a single synchronous browser request. Returning a queued job avoids timeouts and gives the UI something concrete to poll."],
                                ["3. Worker enters StockRadar", "A background worker calls StockRadar.analyze_stock(symbol, mode, period). This orchestration method decides what data must be fetched, which analyzer path should run, and which downstream side effects such as alerts or persistence should happen.", "The API layer stays thin; the business workflow lives in one orchestration entrypoint."],
                                ["4. Data acquisition", "The fetch layer gathers current quote data, historical OHLCV series, fundamentals, news, and social sentiment. Structured data and unstructured text are both collected because later stages use both numeric market state and textual context.", "The LLM is grounded in fetched market data instead of being asked to infer the state of the market from its own prior knowledge."],
                                ["5. Indicator computation", "Historical prices are transformed into RSI, MACD, moving averages, Bollinger bands, price-vs-SMA deltas, and related features. These values are deterministic derivatives of the fetched history, not model output.", "Indicators compress raw time series into signals the prompt, scorer, and downstream logic can use directly."],
                                ["6. RAG lookup", "Before fresh analysis is generated, the analyzer builds a semantic query from the live state, including symbol, mode, RSI, MACD, price, and percentage change. The storage layer embeds that query and calls Supabase RPC functions backed by pgvector to retrieve similar analyses, signals, news, and other stored context.", "This gives the LLM historical precedent and prior context instead of making every decision from a cold start."],
                                ["7. Prompt assembly and LLM inference", "The analyzer merges quote data, indicators, news headlines, social sentiment, and formatted RAG context into a task-specific prompt. LiteLLM then routes the request through the configured provider fallback order until one model returns a valid response.", "This is the actual inference step. The LLM writes the analysis; the embedding system only supplies retrieval context."],
                                ["8. Parsing and guardrails", "The raw model text is parsed as JSON and passed through guardrails that check schema shape, allowed signal values, confidence ceilings, price sanity, and reasoning completeness. If RAG context was used, a validator can also score the quality and grounding of the retrieved context.", "This reduces malformed outputs and obvious hallucinations before the result is treated as product data."],
                                ["9. Post-analysis processing", "Once the analysis passes validation, the app can run algorithmic scoring, risk checks, paper-trading hooks, and alert generation. These steps are deterministic or rule-based layers that enrich or gate the generated analysis.", "The product combines LLM output with explicit trading controls rather than trusting the model alone."],
                                ["10. Persistence and semantic memory", "The final analysis row is stored in Supabase, the analysis text is embedded for future similarity search, and an actionable signal row is written with its own context embedding. Those writes become part of the corpus used by later RAG retrieval.", "Every completed analysis makes the system's memory richer for future runs."],
                                ["11. Polling and render", "The frontend polls the analyze-status endpoint using the returned job id. When the worker finishes, the final payload includes the analysis result and related metadata, which the UI renders into the dashboard.", "Submission and completion are decoupled so the UX stays responsive while the heavy work runs in the background."],
                            ]}
                        />
                        <InfoCard icon={Workflow} title="Why Analysis Is A Background Job" variant="blue">
                            The expensive parts of this path are external I/O and model work: fetching history, retrieving vector matches,
                            waiting for LLM output, and writing multiple DB records. The job pattern isolates those latencies from the browser
                            request and makes retries, polling, and user refreshes much easier to handle.
                        </InfoCard>
                    </SubSection>

                    <SubSection id="chat-walkthrough" title="Chat Request Lifecycle">
                        <p className="text-sm text-muted-foreground mb-4">
                            The chat path is usually synchronous, but it still executes several internal stages before answering. The assistant does not
                            just hand the question straight to an LLM. It restores session context, resolves symbols, gathers direct facts, retrieves
                            semantically similar records, and only then asks the model to produce a response.
                        </p>
                        <DataTable
                            headers={["Stage", "What Happens Technically", "Why It Exists"]}
                            rows={[
                                ["1. Browser -> Next.js chat route", "The browser sends POST /api/ask with a question, optional stock symbol, and optional session id. The route rate-limits the request, trims inputs, rejects empty questions, and proxies the normalized payload to POST /v1/ask.", "This enforces a predictable request shape and prevents obvious bad input from reaching the backend."],
                                ["2. Session restoration", "If a valid session id is supplied, the backend reloads recent conversation turns from chat_history and reconstructs the assistant's in-memory conversation state. If not, it creates a fresh session.", "Multi-turn chat needs continuity. Otherwise the assistant would lose all prior context between requests."],
                                ["3. Symbol grounding and data hydration", "The assistant validates or infers relevant stock symbols and ensures the app has local data for them. If the user asks about AAPL, the system tries to ground the answer in the app's own stored records instead of generic model memory.", "This is what makes the assistant a stock-radar interface rather than a general-purpose chat wrapper."],
                                ["4. Semantic retrieval", "The question is embedded and used to query analyses, signals, news, knowledge entries, and prior conversations through vector similarity. This returns records that are conceptually related, even when their wording does not exactly match the user's phrasing.", "Semantic retrieval is how the assistant finds relevant history and precedent."],
                                ["5. Exact retrieval", "Alongside vector search, the assistant fetches precise structured facts such as the stock row, latest analysis, indicator-bearing records, and recent news by symbol and recency filters.", "Exact lookup is better than semantic similarity when the goal is to surface the freshest known facts without approximation."],
                                ["6. Prompt assembly", "The assistant composes a grounded prompt that includes the user's question, prior session turns, exact factual context, and semantically related context. The prompt deliberately mixes both direct facts and broader retrieved memory.", "The model gets both the latest state and historical context in one place, which leads to more specific answers."],
                                ["7. LLM answer generation", "LiteLLM calls the configured chat model route and returns answer text plus metadata such as model used, token count, and processing time. If the primary provider fails, the fallback order is tried.", "This is the generation step where all the retrieved context is turned into a final answer."],
                                ["8. Persistence of conversation state", "The user turn and assistant turn are both written to Supabase and embedded. That means future chat requests can retrieve prior conversation snippets semantically as part of context building.", "Chat memory becomes searchable product data rather than transient session state."],
                                ["9. Response contract back to the UI", "The backend returns the answer, referenced stock symbols, source summaries, model metadata, session id, and retrieval metrics such as total results and retrieval time.", "The frontend can show not just the answer but also how much context was found and which sources informed it."],
                            ]}
                        />
                        <InfoCard icon={Database} title="Why Chat Uses Exact And Semantic Retrieval" variant="amber">
                            Exact lookup answers questions like &quot;what is the latest stored analysis for AAPL?&quot;.
                            Semantic retrieval answers questions like &quot;what past situations are meaningfully similar to this one?&quot;.
                            The assistant uses both because a finance workflow needs fresh facts and relevant precedent.
                        </InfoCard>
                    </SubSection>
                </Section>

                {/* ═══ 9. CLI ═══ */}
                <Section id="cli" title="CLI Commands Reference" icon={Terminal}>
                    <CodeBlock
                        language="bash"
                        code={`# Analyze a single stock
python main.py analyze AAPL --mode intraday --period max
python main.py analyze RELIANCE.NS --mode longterm --no-alert

# Analyze a user's watchlist
python main.py watchlist user@email.com --mode intraday

# Explain price movement
python main.py explain TSLA

# Continuous monitoring (every 15 min)
python main.py continuous AAPL MSFT GOOGL --interval 15

# AI chat
python main.py chat --symbol AAPL
python main.py ask "Is AAPL a good buy right now?"

# Paper trading
python main.py paper status
python main.py paper trades
python main.py paper dashboard
python main.py paper reset

# Canary / Audit / Backfill
python main.py canary status | enable | disable
python main.py audit report --date 2026-03-07
python main.py backfill AAPL MSFT --period max

# API usage & testing
python main.py usage [--reset]
python main.py test`}
                    />
                </Section>

                {/* ═══ 10. DESIGN DECISIONS ═══ */}
                <Section id="design-decisions" title="Design Decisions & Trade-offs" icon={FileText}>
                    <div className="space-y-4">
                        {[
                            {
                                q: "Why LiteLLM instead of direct API calls?",
                                a: "Single interface for all LLM providers. Switching from Groq to Anthropic requires changing one string, not rewriting API integration code.",
                            },
                            {
                                q: "Why calculate indicators manually?",
                                a: "Understanding: you know exactly what every number means. Portability: no native C library dependency. Transparency: interview-ready — you can explain the math.",
                            },
                            {
                                q: "Why StockScorer alongside the ML model?",
                                a: "The ML model might not be trained yet, or might fail. The scorer provides a reliable, deterministic fallback. It's also more explainable — you can trace exactly why a stock scored 65 on momentum.",
                            },
                            {
                                q: "Why Cohere for embeddings?",
                                a: "Cohere is still the default, but the runtime is now configurable through EMBEDDING_PROVIDER, EMBEDDING_MODEL, and EMBEDDING_DIM. The database schema currently expects 1024-dimensional pgvector columns, so alternative providers should either output 1024-d vectors or be paired with a schema migration.",
                            },
                            {
                                q: "Why paper trading before live?",
                                a: "Professional quant shops never deploy directly to production. Paper trading validates signals actually make money before real capital is at risk.",
                            },
                            {
                                q: "Why kill switches and canary rollouts?",
                                a: "Production-grade safety from Facebook/Google deployment practices. If a model starts making bad trades, kill switch halts everything. Canary tests with a subset first.",
                            },
                        ].map(({ q, a }) => (
                            <div key={q} className="rounded-lg border border-border p-4 bg-card/30">
                                <p className="font-medium text-foreground text-sm mb-2">{q}</p>
                                <p className="text-sm text-muted-foreground">{a}</p>
                            </div>
                        ))}
                    </div>
                </Section>

                {/* Footer */}
                <div className="mt-16 pt-8 border-t border-border text-center">
                    <p className="text-sm text-muted-foreground">
                        Built by <strong className="text-foreground">Darshan</strong> — AI-powered stock analysis
                        bringing institutional-grade intelligence to individual traders.
                    </p>
                    <div className="flex items-center justify-center gap-4 mt-4">
                        <a
                            href="https://github.com/Darshan174/stock-radar"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-green-400 transition-colors"
                        >
                            <ExternalLink className="h-3.5 w-3.5" />
                            View on GitHub
                        </a>
                    </div>
                </div>
            </main>
        </div>
    )
}
