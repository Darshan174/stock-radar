"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import dynamic from "next/dynamic"
import {
    LayoutDashboard,
    TrendingUp,
    Zap,
    BarChart3,
    Settings,
    Radar,
    MessageSquare,
    CreditCard,
    ChevronLeft,
    ChevronRight,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useSidebar } from "@/providers/sidebar-provider"

const ThemeToggle = dynamic(
    () => import("@/components/theme-toggle").then((m) => m.ThemeToggle),
    { ssr: false }
)

const navItems = [
    { href: "/", label: "Dashboard", icon: LayoutDashboard },
    { href: "/stocks", label: "Stocks", icon: TrendingUp },
    { href: "/signals", label: "Signals", icon: Zap },
    { href: "/chat", label: "AI Chat", icon: MessageSquare },
    { href: "/usage", label: "API Usage", icon: BarChart3 },
    { href: "/x402-demo", label: "x402 Demo", icon: CreditCard },
    { href: "/settings", label: "Settings", icon: Settings },
]

export function AppSidebar() {
    const pathname = usePathname()
    const { collapsed, hidden, toggle } = useSidebar()

    if (hidden) return null

    return (
        <aside
            className={cn(
                "fixed left-0 top-0 z-40 h-screen border-r border-sky-200/75 bg-[linear-gradient(180deg,rgba(255,255,255,0.97),rgba(242,248,255,0.95))] backdrop-blur-xl transition-[width] duration-300 ease-out dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(8,33,53,0.92),rgba(5,20,34,0.86))]",
                collapsed ? "w-16" : "w-64"
            )}
        >
            <div className="flex h-full flex-col">
                {/* Logo */}
                <div className="flex h-16 items-center border-b border-sky-200/75 px-3.5 dark:border-white/10">
                    <Link href="/" className="flex min-w-0 items-center gap-2 overflow-hidden">
                        <div className="grid h-7 w-7 shrink-0 place-items-center rounded-md bg-gradient-to-br from-cyan-300 to-lime-300 text-[#06253a] shadow-[0_0_20px_-10px_rgba(84,240,255,0.9)]">
                            <Radar className="h-4 w-4" />
                        </div>
                        {!collapsed && (
                            <span className="bg-gradient-to-r from-sky-700 via-blue-700 to-teal-700 bg-clip-text text-lg font-semibold whitespace-nowrap text-transparent dark:from-cyan-200 dark:via-blue-200 dark:to-lime-200">
                                Stock Radar
                            </span>
                        )}
                    </Link>
                    <button
                        onClick={toggle}
                        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                        className="ml-auto flex h-8 w-8 items-center justify-center rounded-md border border-sky-200/85 bg-white/80 text-slate-600 transition-colors hover:bg-sky-100 hover:text-slate-900 dark:border-white/15 dark:bg-white/5 dark:text-muted-foreground dark:hover:bg-white/10 dark:hover:text-foreground"
                    >
                        {collapsed ? (
                            <ChevronRight className="h-4 w-4" />
                        ) : (
                            <ChevronLeft className="h-4 w-4" />
                        )}
                    </button>
                </div>

                {/* Navigation */}
                <nav className="flex-1 space-y-1.5 p-2.5">
                    {navItems.map((item) => {
                        const isActive = pathname === item.href ||
                            (item.href !== "/" && pathname.startsWith(item.href))

                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                title={collapsed ? item.label : undefined}
                                className={cn(
                                    "flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition-all duration-200",
                                    collapsed && "justify-center px-2",
                                    isActive
                                        ? "border border-sky-300/70 bg-sky-100 text-sky-900 shadow-[0_12px_26px_-20px_rgba(37,99,235,0.35)] dark:border-cyan-300/35 dark:bg-cyan-300/18 dark:text-cyan-100 dark:shadow-[0_12px_26px_-18px_rgba(84,240,255,0.8)]"
                                        : "text-slate-600 hover:bg-sky-100/80 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-white/10 dark:hover:text-white"
                                )}
                            >
                                <item.icon className="h-5 w-5 shrink-0" />
                                {!collapsed && item.label}
                            </Link>
                        )
                    })}
                </nav>

                {/* Footer */}
                <div className="space-y-2 border-t border-sky-200/75 p-2.5 dark:border-white/10">
                    {!collapsed && <ThemeToggle />}
                    {!collapsed && (
                        <p className="px-2 text-[11px] text-slate-500 dark:text-slate-400">
                            Stock Radar v1.0
                        </p>
                    )}
                </div>
            </div>
        </aside>
    )
}
