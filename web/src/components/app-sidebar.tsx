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
                "fixed left-0 top-0 z-40 h-screen border-r border-[#22c55e]/15 bg-black backdrop-blur-xl transition-[width] duration-300 ease-out",
                collapsed ? "w-16" : "w-64"
            )}
        >
            <div className="flex h-full flex-col">
                {/* Logo */}
                <div className="flex h-16 items-center border-b border-[#22c55e]/15 px-3.5">
                    <Link href="/" className="flex min-w-0 items-center gap-2 overflow-hidden">
                        <div className="grid h-7 w-7 shrink-0 place-items-center rounded-md bg-gradient-to-br from-[#22c55e] to-[#4ade80] text-black shadow-[0_0_20px_-10px_rgba(34,197,94,0.9)]">
                            <Radar className="h-4 w-4" />
                        </div>
                        {!collapsed && (
                            <span className="text-lg font-semibold whitespace-nowrap text-white">
                                Stock Radar
                            </span>
                        )}
                    </Link>
                    <button
                        onClick={toggle}
                        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                        className="ml-auto flex h-8 w-8 items-center justify-center rounded-md border border-[#22c55e]/15 bg-white/5 text-neutral-400 transition-colors hover:bg-[#22c55e]/10 hover:text-white"
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
                                        ? "border border-[#22c55e]/35 bg-[#22c55e]/15 text-[#4ade80] shadow-[0_12px_26px_-18px_rgba(34,197,94,0.6)]"
                                        : "text-neutral-400 hover:bg-white/5 hover:text-white"
                                )}
                            >
                                <item.icon className="h-5 w-5 shrink-0" />
                                {!collapsed && item.label}
                            </Link>
                        )
                    })}
                </nav>

                {/* Footer */}
                <div className="space-y-2 border-t border-[#22c55e]/15 p-2.5">
                    {!collapsed && <ThemeToggle />}
                    {!collapsed && (
                        <p className="px-2 text-[11px] text-neutral-500">
                            Stock Radar v1.0
                        </p>
                    )}
                </div>
            </div>
        </aside>
    )
}
