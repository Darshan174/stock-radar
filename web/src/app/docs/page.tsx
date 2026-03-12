"use client"

import { useState } from "react"
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
    MessageSquare,
    TrendingUp,
    BarChart3,
    Bell,
    Search,
    Settings,
    Gauge,
    DollarSign,
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
    if (typeof window !== "undefined") {
        // eslint-disable-next-line react-hooks/rules-of-hooks
        const { useEffect } = require("react")
        // eslint-disable-next-line react-hooks/rules-of-hooks
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
    }

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
                            code={`┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACES                        │
│  Next.js Dashboard  │  CLI Terminal  │  Slack/Telegram Bots │
│         │                  │                  ▲              │
│         ▼                  ▼                  │              │
│     FastAPI            main.py ───────────────┘              │
│     Backend            (CLI)                                 │
│         └────────┬────────┘                                  │
│                  ▼                                            │
│  ┌─────────────────────────────────────────────────────┐     │
│  │          STOCK RADAR — ORCHESTRATOR                  │     │
│  │  1. Fetch → 2. Analyze → 3. Verify → 4. Store → 5. Alert │
│  └──────┬──────────┬──────────┬──────────┬──────────┘  │     │
│         │          │          │          │              │     │
│   StockFetcher  StockAnalyzer  Storage  Notifications  │     │
│   •TwelveData   •LLM Chain    •Supabase  •Slack        │     │
│   •yFinance     •ML Model     •Cohere    •Telegram     │     │
│   •Finnhub      •Scorer       •pgvector                │     │
│   •Reddit API   •RAG                                   │     │
│                                                         │     │
│  ┌──── SUPPORTING ─────────────────────────────────┐   │     │
│  │ Config │ Cache │ Guardrails │ Metrics │ Prompts  │   │     │
│  └─────────────────────────────────────────────────┘   │     │
│                                                         │     │
│  ┌──── TRAINING / ML ──────────────────────────────┐   │     │
│  │ Predictor │ Paper Trading │ Broker │ Risk │ Kill │   │     │
│  └─────────────────────────────────────────────────┘   │     │
└─────────────────────────────────────────────────────────────┘`}
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
                                    desc: "Retrieves RAG context (similar past analyses, signals, news by vector similarity). Builds a structured prompt — intraday includes social sentiment & real-time buzz; longterm includes 50+ fundamentals (valuation, profitability, growth, health, dividends, analyst consensus). Calls the LLM with fallback chain. RAG Validation grades the output (A-F) on faithfulness, relevancy, groundedness & temporal validity.",
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
                                    desc: "Saves to Supabase: stock record, price data, indicators, analysis with vector embedding (Cohere), and actionable signals for tracking.",
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
                                ["analyses", "AI analysis results + embeddings", "✓ pgvector"],
                                ["signals", "Actionable trading signals", "No"],
                                ["alerts", "Notification delivery records", "No"],
                                ["news", "Stored news articles + embeddings", "✓ pgvector"],
                                ["knowledge", "RAG knowledge base", "✓ pgvector"],
                            ]}
                        />
                        <InfoCard icon={Search} title="Vector Embeddings — Semantic Search" variant="blue">
                            When storing an analysis, Cohere generates a 1024-dimensional vector representation of its meaning.
                            Later, when asking &quot;show me stocks with similar patterns&quot;, the system finds analyses that are
                            <em> meaningfully similar</em> — not just keyword-matching. This powers the RAG system.
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
                            — RAG-powered conversational assistant. Before answering, it searches the database for relevant
                            historical analyses, signals, and news to ground its response in real data.
                        </p>
                    </SubSection>

                    {/* 4.7 RAG */}
                    <SubSection id="rag" title="RAG System — Retriever & Validator">
                        <p className="text-muted-foreground text-sm mb-4">
                            <strong className="text-foreground">RAGRetriever</strong>{" "}
                            <code className="text-green-400 bg-green-500/10 px-1 rounded text-xs">rag_retriever.py</code> — Searches
                            similar past analyses, historical signals, related news, and correlated stock signals by vector similarity.
                        </p>
                        <p className="text-muted-foreground text-sm mb-4">
                            <strong className="text-foreground">RAGValidator</strong>{" "}
                            <code className="text-green-400 bg-green-500/10 px-1 rounded text-xs">rag_validator.py</code> — RAGAS-style
                            metrics: Faithfulness, Context Relevancy, Groundedness, Temporal Validity. Each analysis gets a letter grade (A-F).
                        </p>
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
                        Auth via Bearer token (<code className="text-green-400 text-xs">BACKEND_AUTH_TOKEN</code>) on all <code className="text-green-400 text-xs">/v1/</code> routes.
                        Long-running analyses execute in background thread pools (max_workers=2, 30-min TTL).
                        Analysis payloads now include a <code className="text-green-400 text-xs">guardrails</code> key with pass/fail status and issue details.
                    </p>
                </Section>

                {/* ═══ 8. PIPELINE WALKTHROUGH ═══ */}
                <Section id="pipeline-walkthrough" title="Full Pipeline Walkthrough" icon={GitBranch}>
                    <p className="text-muted-foreground text-sm mb-4">
                        Here&apos;s the complete journey when a user enters &quot;AAPL&quot; in the dashboard:
                    </p>
                    <CodeBlock
                        language="text"
                        code={`1. Frontend → POST /v1/analyze {symbol: "AAPL", mode: "intraday"}
2. Backend creates background job, returns job_id
3. StockRadar.analyze_stock("AAPL") called
4. _resolve_analysis_history_config("intraday") → 5d, 15m bars
5. PARALLEL FETCH (up to 6 workers):
   • WebSocket cache check → instant if < 60s fresh
   • get_quote → StockQuote
   • get_price_history("5d","15m") → [PriceData × 960]
   • get_fundamentals → {50+ metrics: valuation, growth, ...}
   • get_news_yahoo → [NewsItem × 10]
   • get_news_finnhub → [NewsItem × 15]
   • get_sentiment_finnhub → {bullish:30, bearish:5, ...}
6. calculate_indicators → RSI=62, MACD=15.5, SMA20=2800
7. get_social_sentiment → {reddit_mentions: 342}
8. analyzer.analyze_intraday():
   a. RAG context: past analyses + signals + news (vector similarity)
   b. Build prompt: price + technicals + news + social + RAG
   c. LLM call: Groq Llama 70B → ✓ 1.2s
   d. Parse JSON → Signal=BUY, Confidence=0.75
   e. RAG Validation → Faithfulness=0.85, Grade=B+, Temporal=✓
9. generate_algo_prediction():
   a. RegimeAwarePredictor → regime-specific model
      predict(indicators + OHLCV + headlines + timestamps
              + finnhub_sentiment) [37+ features]
      → signal=buy, confidence=0.72
   b. StockScorer → M=65, V=55, Q=70, R=4
      (with per-indicator score_breakdowns)
   c. Market regime → "neutral" (confidence=0.6)
   d. Position size → 2.5% of portfolio
   e. Stop/Take (ATR) → SL=$178.50, TP=$195.00
   f. Per-trade risk → 1.8% ✓ within 2% limit
10. Paper trading (kill switch → canary → risk → order)
11. Storage: analysis + embedding saved to Supabase
12. Alerts: Slack + Telegram notifications sent
13. Frontend polls job → renders signal, scores, breakdowns`}
                    />
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
                                a: "High-quality 1024-dim vectors, generous free tier (1,000 calls/month), and specifically designed for search/retrieval use cases.",
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
