"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
    LayoutDashboard,
    TrendingUp,
    Zap,
    BarChart3,
    Settings,
    LineChart,
    MessageSquare,
    CreditCard,
    ChevronLeft,
    ChevronRight,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { ThemeToggle } from "@/components/theme-toggle"
import { useSidebar } from "@/providers/sidebar-provider"

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
                "fixed left-0 top-0 z-40 h-screen border-r bg-background transition-[width] duration-200",
                collapsed ? "w-16" : "w-64"
            )}
        >
            {/* Edge collapse/expand button */}
            <button
                onClick={toggle}
                title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                className="absolute -right-3 top-1/2 -translate-y-1/2 z-50 flex h-6 w-6 items-center justify-center rounded-full border bg-background shadow-sm hover:bg-muted transition-colors"
            >
                {collapsed ? (
                    <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                ) : (
                    <ChevronLeft className="h-3.5 w-3.5 text-muted-foreground" />
                )}
            </button>

            <div className="flex h-full flex-col">
                {/* Logo */}
                <div className="flex h-16 items-center border-b px-4">
                    <Link href="/" className="flex items-center gap-2 overflow-hidden">
                        <LineChart className="h-6 w-6 shrink-0 text-primary" />
                        {!collapsed && (
                            <span className="text-xl font-bold whitespace-nowrap">
                                Stock Radar
                            </span>
                        )}
                    </Link>
                </div>

                {/* Navigation */}
                <nav className="flex-1 space-y-1 p-2">
                    {navItems.map((item) => {
                        const isActive = pathname === item.href ||
                            (item.href !== "/" && pathname.startsWith(item.href))

                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                title={collapsed ? item.label : undefined}
                                className={cn(
                                    "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                                    collapsed && "justify-center px-2",
                                    isActive
                                        ? "bg-primary text-primary-foreground"
                                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                                )}
                            >
                                <item.icon className="h-5 w-5 shrink-0" />
                                {!collapsed && item.label}
                            </Link>
                        )
                    })}
                </nav>

                {/* Footer */}
                <div className="border-t p-2 space-y-2">
                    {!collapsed && <ThemeToggle />}
                    {!collapsed && (
                        <p className="text-xs text-muted-foreground px-3">
                            Stock Radar v1.0
                        </p>
                    )}
                </div>
            </div>
        </aside>
    )
}
