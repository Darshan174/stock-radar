"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import {
  ArrowRight,
  Radar,
  Zap,
  Shield,
  Brain,
  Activity,
  Database,
  MessageSquare,
  BarChart3,
  TrendingUp,
  Wallet,
  Globe,
  Cpu,
  Layers,
  GitBranch,
  Radio,
  LineChart,
  Gauge,
  FlaskConical,
  Search,
  Workflow,
  Crosshair,
  Sparkles,
  ShieldCheck,
  Target,
  TestTube2,
} from "lucide-react"
import { Space_Grotesk, IBM_Plex_Mono } from "next/font/google"
import styles from "./project-intro.module.css"

const displayFont = Space_Grotesk({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-display-intro",
})

const monoFont = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono-intro",
})

// ─── Typewriter Effect ────────────────────────────────────────
const TYPEWRITER_PHRASES = [
  "Financial Intelligence",
  "Autonomous Agent",
  "Micropayment Protocol",
  "On-Chain Identity",
  "AI-Powered Analysis",
]

function useTypewriter(phrases: string[], typingSpeed = 80, deletingSpeed = 40, pauseTime = 2000) {
  const [text, setText] = useState("")
  const [phraseIndex, setPhraseIndex] = useState(0)
  const [isDeleting, setIsDeleting] = useState(false)

  useEffect(() => {
    const currentPhrase = phrases[phraseIndex]

    const timeout = setTimeout(
      () => {
        if (!isDeleting) {
          setText(currentPhrase.slice(0, text.length + 1))
          if (text.length + 1 === currentPhrase.length) {
            setTimeout(() => setIsDeleting(true), pauseTime)
          }
        } else {
          setText(currentPhrase.slice(0, text.length - 1))
          if (text.length === 0) {
            setIsDeleting(false)
            setPhraseIndex((prev) => (prev + 1) % phrases.length)
          }
        }
      },
      isDeleting ? deletingSpeed : typingSpeed
    )

    return () => clearTimeout(timeout)
  }, [text, phraseIndex, isDeleting, phrases, typingSpeed, deletingSpeed, pauseTime])

  return text
}

// ─── Particle System ──────────────────────────────────────────
interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  color: string
  alpha: number
  decay: number
}

function useParticles(canvasRef: React.RefObject<HTMLCanvasElement | null>) {
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let animationId: number
    const particles: Particle[] = []
    const COLORS = [
      "159, 232, 112",  // wise green
      "125, 217, 87",   // darker wise green
      "184, 233, 134",  // light lime
      "78, 201, 176",   // teal-green
      "200, 247, 166",  // pale lime
      "104, 211, 145",  // mid green
      "232, 245, 163",  // warm lime
    ]

    function resize() {
      canvas!.width = window.innerWidth
      canvas!.height = window.innerHeight
    }
    resize()
    window.addEventListener("resize", resize)

    // Seed initial particles
    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        radius: Math.random() * 2.2 + 0.6,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
        alpha: Math.random() * 0.5 + 0.15,
        decay: 0,
      })
    }

    function draw() {
      ctx!.clearRect(0, 0, canvas!.width, canvas!.height)

      for (const p of particles) {
        p.x += p.vx
        p.y += p.vy

        // Wrap around
        if (p.x < 0) p.x = canvas!.width
        if (p.x > canvas!.width) p.x = 0
        if (p.y < 0) p.y = canvas!.height
        if (p.y > canvas!.height) p.y = 0

        // Subtle pulsation
        const pulse = Math.sin(Date.now() * 0.001 + p.x * 0.01) * 0.15 + 0.85

        ctx!.beginPath()
        ctx!.arc(p.x, p.y, p.radius * pulse, 0, Math.PI * 2)
        ctx!.fillStyle = `rgba(${p.color}, ${p.alpha * pulse})`
        ctx!.fill()

        // Glow
        ctx!.beginPath()
        ctx!.arc(p.x, p.y, p.radius * 3 * pulse, 0, Math.PI * 2)
        ctx!.fillStyle = `rgba(${p.color}, ${p.alpha * 0.08 * pulse})`
        ctx!.fill()
      }

      // Draw faint connection lines between nearby particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 140) {
            ctx!.beginPath()
            ctx!.moveTo(particles[i].x, particles[i].y)
            ctx!.lineTo(particles[j].x, particles[j].y)
            ctx!.strokeStyle = `rgba(148, 163, 184, ${0.04 * (1 - dist / 140)})`
            ctx!.lineWidth = 0.5
            ctx!.stroke()
          }
        }
      }

      animationId = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener("resize", resize)
    }
  }, [canvasRef])
}

// ─── Scroll Reveal Hook ──────────────────────────────────────
function useReveal() {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const node = ref.current
    if (!node) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          observer.unobserve(node)
        }
      },
      { threshold: 0.15 }
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  return { ref, className: `${styles.reveal} ${visible ? styles.revealVisible : ""}` }
}

// ─── Feature Data ────────────────────────────────────────────
const FEATURES = [
  {
    icon: Shield,
    color: "Cyan",
    title: "x402 Payment Protocol",
    desc: "HTTP 402 Payment Required — the internet's forgotten status code, implemented for real. Any HTTP client that can read a 402 response and submit an Aptos transaction can use the API. No SDK required, no registration, no friction.",
  },
  {
    icon: Wallet,
    color: "Teal",
    title: "On-Chain Agent Identity",
    desc: "Registered as a verified entity on Aptos testnet with 10 declared capabilities. Move smart contracts track total requests, successes, and earnings — a tamper-proof reputation that other agents can verify before paying.",
  },
  {
    icon: Activity,
    color: "Green",
    title: "Real-Time Data Pipeline",
    desc: "WebSocket-powered live price feeds with sub-second OHLC candle and volume updates. TradingView-style charts rendered via uPlot and lightweight-charts with real-time streaming from market data providers.",
  },
  {
    icon: MessageSquare,
    color: "Blue",
    title: "Agent-to-Agent Economy",
    desc: "XMTP-compatible HTTP messaging enables capability discovery, pricing negotiation, and automated task delegation. Agents discover, negotiate, pay, and rate each other without human intermediaries.",
  },
  {
    icon: Layers,
    color: "Gold",
    title: "Task Orchestration Engine",
    desc: "Parallel sub-analysis execution via /orchestrate — momentum (40%), fundamentals (40%), and sentiment (20%) combined into a weighted aggregate. Priced at 400 octas (discount vs. individual calls).",
  },
  {
    icon: Database,
    color: "Purple",
    title: "Move Smart Contracts",
    desc: "Three Move modules deployed on Aptos: minimal_registry for agent registration & capabilities, agent_registry for extended identity with ratings & tags, and agent_marketplace for task lifecycle, escrow, and dispute resolution.",
  },
]

// ─── AI/ML Deep Dive ─────────────────────────────────────────
const AI_ML_FEATURES = [
  {
    icon: Brain,
    color: "Purple",
    title: "Multi-Model LLM Pipeline",
    desc: "Production fallback chain with GLM-5 (primary) → Gemini (secondary). LiteLLM abstraction enables seamless model switching. Structured JSON extraction from LLM responses powers all trading signal generation.",
    details: [
      "ZAI GLM-5 OpenAI-compatible endpoint as primary",
      "Google Gemini automatic fallback on failure",
      "Token accounting & usage tracking per model",
      "System + user prompt engineering per analysis mode",
    ],
  },
  {
    icon: Gauge,
    color: "Cyan",
    title: "Algorithmic Scoring Engine",
    desc: "Formula-based, interpretable scoring across four dimensions — Momentum, Value, Quality, and Risk — with configurable weight presets (balanced, momentum_focus, value_focus, quality_focus, conservative).",
    details: [
      "Momentum: RSI zones, MACD crossovers, price vs. SMA",
      "Value: P/E, P/B, P/S ratios, dividend yield analysis",
      "Quality: ROE, profit margins, debt-to-equity, current ratio",
      "Risk: ATR volatility, ADX trend strength, debt levels",
    ],
  },
  {
    icon: Search,
    color: "Teal",
    title: "RAG-Augmented Analysis",
    desc: "Retrieval-Augmented Generation injects real-time market context into LLM prompts. A dedicated RAG Retriever fetches relevant earnings data, sector news, and historical patterns to ground AI predictions in facts.",
    details: [
      "RAG Retriever for domain-specific context injection",
      "RAG Validator cross-checks AI outputs against data",
      "Quality badges: Verified, Grounded, Speculative",
      "Source attribution for every AI-generated claim",
    ],
  },
  {
    icon: Sparkles,
    color: "Gold",
    title: "FinBERT Sentiment Analysis",
    desc: "Multi-source sentiment pipeline combining FinBERT transformer model for financial text classification, Finnhub social sentiment scores, and earnings call analysis for comprehensive market mood assessment.",
    details: [
      "FinBERT fine-tuned on financial corpus",
      "News headline impact scoring & classification",
      "Social media sentiment aggregation via Finnhub",
      "Earnings sentiment integration into ML features",
    ],
  },
  {
    icon: FlaskConical,
    color: "Green",
    title: "Feature Engineering Pipeline",
    desc: "90+ engineered features for the ML prediction model — from raw price/volume data through technical indicators, cross-sectional features, and regime-aware transformations, all built by the dataset builder.",
    details: [
      "Technical: RSI, MACD, Bollinger Bands, ATR, ADX",
      "Cross-sectional: sector relative strength, peer ranking",
      "Temporal: lagged returns, rolling volatility, momentum",
      "Regime-aware: feature scaling per market regime",
    ],
  },
  {
    icon: Crosshair,
    color: "Blue",
    title: "Regime Detection & Routing",
    desc: "Hidden Markov Model-based market regime classification (bull / bear / sideways / volatile). The regime router dynamically adjusts model weights, scoring presets, and risk parameters based on detected market conditions.",
    details: [
      "HMM-based regime classification from price data",
      "Regime-specific model weight adjustment",
      "Dynamic risk parameter scaling per regime",
      "Canary deployment: new models tested per-regime",
    ],
  },
  {
    icon: Target,
    color: "Cyan",
    title: "ML Predictor & Model Registry",
    desc: "Gradient-boosted tree predictor trained on historical stock data with walk-forward validation. A model registry manages versioning, A/B testing, and canary deployments with automatic rollback on performance degradation.",
    details: [
      "XGBoost / LightGBM gradient-boosted ensemble",
      "Walk-forward cross-validation, no look-ahead bias",
      "Model registry with version tracking & metadata",
      "Canary deployments with kill-switch on drift",
    ],
  },
  {
    icon: TestTube2,
    color: "Gold",
    title: "Backtesting & Paper Trading",
    desc: "Full portfolio-level backtesting engine with realistic constraints — slippage, position limits, sector exposure caps. Paper trading simulator runs live signals against real market data to validate before deployment.",
    details: [
      "Portfolio backtesting with constraints & slippage",
      "Paper trading with live market data validation",
      "Pre-trade risk checks & position sizing rules",
      "Calibration module: predicted vs. realized returns",
    ],
  },
  {
    icon: ShieldCheck,
    color: "Teal",
    title: "ML Ops & Guardrails",
    desc: "Production-grade ML operations with feature health monitoring, data drift detection, kill switches for anomalous predictions, and an audit trail. LLM guardrails prevent hallucinated or dangerous trading advice.",
    details: [
      "Feature health monitoring & data drift alerts",
      "Kill switches on anomalous model outputs",
      "LLM guardrails: hallucination & safety checks",
      "Audit trail for every prediction & decision",
    ],
  },
]

const PRICING = [
  { endpoint: "/live-price", price: "50", unit: "octas", desc: "Real-time price, volume, OHLC" },
  { endpoint: "/momentum", price: "100", unit: "octas", desc: "RSI, MACD, volume signals" },
  { endpoint: "/stock-score", price: "200", unit: "octas", desc: "Multi-factor algo scoring" },
  { endpoint: "/orchestrate", price: "400", unit: "octas", desc: "All analyses combined" },
  { endpoint: "/analyze", price: "500", unit: "octas", desc: "Full AI + LLM reasoning" },
  { endpoint: "/sentiment", price: "100", unit: "octas", desc: "FinBERT + social sentiment" },
  { endpoint: "/fundamentals", price: "100", unit: "octas", desc: "P/E, ROE, margins, targets" },
  { endpoint: "/news-impact", price: "150", unit: "octas", desc: "News sentiment & price impact" },
]

const TECH_STACK = [
  { label: "Next.js 15", icon: Globe },
  { label: "React 19", icon: Cpu },
  { label: "Tailwind CSS 4", icon: Layers },
  { label: "Aptos Blockchain", icon: Database },
  { label: "Move Language", icon: GitBranch },
  { label: "GLM-5 / Gemini", icon: Brain },
  { label: "Supabase", icon: Database },
  { label: "XMTP Protocol", icon: MessageSquare },
  { label: "Petra Wallet", icon: Wallet },
  { label: "WebSocket", icon: Radio },
  { label: "XGBoost / LightGBM", icon: TrendingUp },
  { label: "FinBERT", icon: Sparkles },
  { label: "LiteLLM", icon: Workflow },
  { label: "Radix UI", icon: Layers },
]

// ─── Main Component ──────────────────────────────────────────
interface ProjectIntroProps {
  onEnter: () => void
}

export function ProjectIntro({ onEnter }: ProjectIntroProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const typedText = useTypewriter(TYPEWRITER_PHRASES)
  useParticles(canvasRef)

  // Scroll-reveal refs
  const showcase = useReveal()
  const features = useReveal()
  const aiml = useReveal()
  const pricing = useReveal()
  const tech = useReveal()
  const cta = useReveal()

  // Smooth scroll helper
  const scrollTo = useCallback((id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" })
  }, [])

  return (
    <div className={`${styles.introRoot} ${displayFont.variable} ${monoFont.variable}`}>
      {/* Particle canvas */}
      <canvas ref={canvasRef} className={styles.particleCanvas} />

      {/* Background mesh gradient */}
      <div className={styles.bgMesh} aria-hidden="true" />

      {/* ── Navigation ── */}
      <nav className={styles.nav}>
        <div className={styles.navLogo}>
          <div className={styles.navLogoIcon}>
            <Radar size={16} />
          </div>
          Stock Radar
        </div>
        <div className={styles.navLinks}>
          <button className={styles.navLink} onClick={() => scrollTo("features")}>Features</button>
          <button className={styles.navLink} onClick={() => scrollTo("aiml")}>AI / ML</button>
          <button className={styles.navLink} onClick={() => scrollTo("pricing")}>Pricing</button>
          <button className={styles.navLink} onClick={() => scrollTo("tech")}>Tech Stack</button>
        </div>
        <button className={styles.navCta} onClick={onEnter}>
          Enter Dashboard
        </button>
      </nav>

      {/* ── Hero ── */}
      <section className={styles.hero}>
        <div className={styles.heroBadge}>
          <span className={styles.heroBadgeDot} />
          Autonomous Agent on Aptos
        </div>

        <h1 className={styles.heroTitle}>
          <span className={styles.heroTitleLine}>Experience liftoff with</span>
          <span className={styles.heroTitleLine}>
            <span className={styles.typewriterWrap}>
              <span className={styles.heroTitleGradient}>{typedText}</span>
              <span className={styles.typewriterCursor} />
            </span>
          </span>
        </h1>

        <p className={styles.heroSubtitle}>
          Pay-per-use AI stock analysis powered by x402 micropayments, on-chain identity, and
          agent-to-agent coordination. No accounts. No subscriptions. Just micropayments.
        </p>

        <div className={styles.heroActions}>
          <button className={styles.btnPrimary} onClick={onEnter}>
            Enter Dashboard
            <ArrowRight size={18} />
          </button>
          <button className={styles.btnSecondary} onClick={() => scrollTo("features")}>
            Explore Features
            <Activity size={16} />
          </button>
        </div>
      </section>



      {/* ── IDE Showcase ── */}
      <section className={styles.showcaseSection} ref={showcase.ref}>
        <div className={`${styles.showcaseWindow} ${showcase.className}`}>
          <div className={styles.showcaseWindowBar}>
            <div className={`${styles.windowDot} ${styles.windowDotRed}`} />
            <div className={`${styles.windowDot} ${styles.windowDotYellow}`} />
            <div className={`${styles.windowDot} ${styles.windowDotGreen}`} />
            <span className={styles.windowTitle}>stock-radar — x402 Payment Flow</span>
          </div>
          <div className={styles.showcaseContent}>
            {/* Code Side */}
            <div className={styles.showcaseLeft}>
              <div className={styles.codeBlock}>
                <span className={styles.codeLine}>
                  <span className={styles.codeComment}>{"// x402 Payment Verification"}</span>
                </span>
                <span className={styles.codeLine}>
                  <span className={styles.codeKeyword}>const</span>{" "}
                  <span className={styles.codeFunc}>verifyPayment</span>{" "}
                  <span className={styles.codeOperator}>= </span>
                  <span className={styles.codeKeyword}>async</span>{" "}
                  <span className={styles.codeOperator}>{"(txHash) => {"}</span>
                </span>
                <span className={styles.codeLine}>
                  {"  "}
                  <span className={styles.codeKeyword}>const</span> tx{" "}
                  <span className={styles.codeOperator}>=</span>{" "}
                  <span className={styles.codeKeyword}>await</span>{" "}
                  <span className={styles.codeFunc}>aptosClient</span>
                  <span className={styles.codeOperator}>.</span>
                  <span className={styles.codeFunc}>getTransactionByHash</span>(txHash)
                </span>
                <span className={styles.codeLine}>
                  {"  "}
                  <span className={styles.codeKeyword}>if</span> (tx.success{" "}
                  <span className={styles.codeOperator}>&&</span> tx.amount{" "}
                  <span className={styles.codeOperator}>{"≥"}</span>{" "}
                  <span className={styles.codeNumber}>100</span>) {"{"}
                </span>
                <span className={styles.codeLine}>
                  {"    "}
                  <span className={styles.codeKeyword}>await</span>{" "}
                  <span className={styles.codeFunc}>updateReputation</span>
                  <span className={styles.codeOperator}>(</span>
                  <span className={styles.codeString}>&quot;success&quot;</span>
                  <span className={styles.codeOperator}>)</span>
                </span>
                <span className={styles.codeLine}>
                  {"    "}
                  <span className={styles.codeKeyword}>return</span>{" "}
                  <span className={styles.codeFunc}>serveAnalysis</span>
                  <span className={styles.codeOperator}>(symbol)</span>
                </span>
                <span className={styles.codeLine}>{"  }"}</span>
                <span className={styles.codeLine}>
                  {"  "}
                  <span className={styles.codeKeyword}>throw</span>{" "}
                  <span className={styles.codeKeyword}>new</span>{" "}
                  <span className={styles.codeFunc}>PaymentError</span>(
                  <span className={styles.codeNumber}>402</span>)
                </span>
                <span className={styles.codeLine}>{"}"}</span>
              </div>
            </div>

            {/* Terminal + Agent side */}
            <div className={styles.showcaseRight}>
              <div className={styles.terminalBlock}>
                <div className={styles.terminalPrompt}>
                  <span className={styles.terminalGreen}>$</span>{" "}
                  <span className={styles.terminalCyan}>curl</span>{" "}
                  -X POST /api/agent/momentum
                  <br />
                  <span className={styles.terminalGold}>→ 402 Payment Required</span>
                  <br />
                  {"  amount: "}
                  <span className={styles.terminalCyan}>100 octas</span>
                  {"  deadline: "}
                  <span className={styles.terminalCyan}>30s</span>
                  <br />
                  <span className={styles.terminalGreen}>$</span> Pay → Retry →{" "}
                  <span className={styles.terminalGreen}>✓ 200 OK</span>
                </div>
              </div>

              <div className={styles.agentCard}>
                <div className={styles.agentCardHeader}>
                  <div className={styles.agentCardTitle}>
                    <Radar size={14} />
                    Agent Status
                  </div>
                  <span className={styles.agentPill}>LIVE</span>
                </div>
                <div className={styles.agentStats}>
                  <div className={styles.agentStat}>
                    <span className={styles.agentStatLabel}>Requests</span>
                    <span className={styles.agentStatValue}>14,283</span>
                  </div>
                  <div className={styles.agentStat}>
                    <span className={styles.agentStatLabel}>Success</span>
                    <span className={styles.agentStatValue}>99.7%</span>
                  </div>
                  <div className={styles.agentStat}>
                    <span className={styles.agentStatLabel}>Capabilities</span>
                    <span className={styles.agentStatValue}>10</span>
                  </div>
                  <div className={styles.agentStat}>
                    <span className={styles.agentStatLabel}>Revenue</span>
                    <span className={styles.agentStatValue}>2.41 APT</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section id="features" className={styles.featuresSection} ref={features.ref}>
        <div className={`${styles.sectionHeader} ${features.className}`}>
          <div className={styles.sectionKicker}>
            <Zap size={13} />
            Core Capabilities
          </div>
          <h2 className={styles.sectionTitle}>Built for the agent-first era</h2>
          <p className={styles.sectionSubtitle}>
            Stock Radar is a verified autonomous agent registered on the Aptos blockchain, selling
            financial intelligence through x402 micropayments.
          </p>
        </div>

        <div className={`${styles.featuresGrid} ${features.className}`}>
          {FEATURES.map((f) => (
            <div key={f.title} className={styles.featureCard}>
              <div className={`${styles.featureIcon} ${styles[`featureIcon${f.color}` as keyof typeof styles]}`}>
                <f.icon size={22} />
              </div>
              <div className={styles.featureTitle}>{f.title}</div>
              <div className={styles.featureDesc}>{f.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── AI/ML Deep Dive ── */}
      <section id="aiml" className={styles.featuresSection} ref={aiml.ref}>
        <div className={`${styles.sectionHeader} ${aiml.className}`}>
          <div className={styles.sectionKicker}>
            <Brain size={13} />
            AI / ML Engineering
          </div>
          <h2 className={styles.sectionTitle}>
            Serious ML, not just a{" "}
            <span className={styles.heroTitleGradient}>wrapper</span>
          </h2>
          <p className={styles.sectionSubtitle}>
            A production-grade ML pipeline — from multi-model LLM orchestration and feature
            engineering through regime-aware prediction, backtesting, and real-time guardrails.
          </p>
        </div>

        <div className={`${styles.aimlGrid} ${aiml.className}`}>
          {AI_ML_FEATURES.map((f) => (
            <div key={f.title} className={styles.aimlCard}>
              <div className={styles.aimlCardTop}>
                <div className={`${styles.featureIcon} ${styles[`featureIcon${f.color}` as keyof typeof styles]}`}>
                  <f.icon size={22} />
                </div>
                <div>
                  <div className={styles.featureTitle}>{f.title}</div>
                  <div className={styles.featureDesc}>{f.desc}</div>
                </div>
              </div>
              <ul className={styles.aimlDetails}>
                {f.details.map((d) => (
                  <li key={d} className={styles.aimlDetail}>
                    <span className={styles.aimlBullet}>→</span>
                    {d}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* ── Pricing ── */}
      <section id="pricing" className={styles.pricingSection} ref={pricing.ref}>
        <div className={`${styles.pricingStrip} ${pricing.className}`}>
          <div className={styles.pricingHeader}>
            <div>
              <div className={styles.pricingTitle}>Micro-Pricing</div>
              <div className={styles.pricingSubtitle}>
                All prices in octas (1 APT = 100,000,000 octas). Discovery & messaging are free.
              </div>
            </div>
          </div>
          <div className={styles.pricingGrid}>
            {PRICING.map((p) => (
              <div key={p.endpoint} className={styles.pricingItem}>
                <div className={styles.pricingEndpoint}>{p.endpoint}</div>
                <div className={styles.pricingAmount}>
                  {p.price}
                  <span className={styles.pricingAmountUnit}>{p.unit}</span>
                </div>
                <div className={styles.pricingDesc}>{p.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Tech Stack ── */}
      <section id="tech" className={styles.techSection} ref={tech.ref}>
        <div className={`${styles.sectionHeader} ${tech.className}`}>
          <div className={styles.sectionKicker}>
            <Cpu size={13} />
            Technology
          </div>
          <h2 className={styles.sectionTitle}>Powered by cutting-edge tech</h2>
          <p className={styles.sectionSubtitle}>
            A modern stack combining Next.js, Aptos blockchain, and AI/ML for seamless financial intelligence.
          </p>
        </div>
        <div className={`${styles.techGrid} ${tech.className}`}>
          {TECH_STACK.map((t) => (
            <div key={t.label} className={styles.techPill}>
              <t.icon size={16} />
              {t.label}
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section className={styles.ctaSection} ref={cta.ref}>
        <div className={`${styles.ctaGlow} ${cta.className}`}>
          <h2 className={styles.ctaTitle}>
            Ready for <span className={styles.heroTitleGradient}>liftoff</span>?
          </h2>
          <p className={styles.ctaSubtitle}>
            Experience Stock Radar&apos;s autonomous financial intelligence —
            pay-per-use AI analysis with zero friction.
          </p>
          <button className={styles.btnPrimary} onClick={onEnter}>
            Enter Dashboard
            <ArrowRight size={18} />
          </button>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <p className={styles.footerText}>
          Built with
          <span className={styles.footerAccent}>♦</span>
          Stock Radar — Autonomous Financial Intelligence Agent on Aptos
        </p>
      </footer>
    </div>
  )
}
