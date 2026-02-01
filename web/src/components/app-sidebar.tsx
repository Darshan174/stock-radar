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
    Search,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { ThemeToggle } from "@/components/theme-toggle"

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

    return (
        <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r bg-background">
            <div className="flex h-full flex-col">
                {/* Logo */}
                <div className="flex h-16 items-center border-b px-6">
                    <Link href="/" className="flex items-center gap-2">
                        <LineChart className="h-6 w-6 text-primary" />
                        <span className="text-xl font-bold">Stock Radar</span>
                    </Link>
                </div>

                {/* Navigation */}
                <nav className="flex-1 space-y-1 p-4">
                    {navItems.map((item) => {
                        const isActive = pathname === item.href ||
                            (item.href !== "/" && pathname.startsWith(item.href))

                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={cn(
                                    "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                                    isActive
                                        ? "bg-primary text-primary-foreground"
                                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                                )}
                            >
                                <item.icon className="h-5 w-5" />
                                {item.label}
                            </Link>
                        )
                    })}
                </nav>

                {/* Footer with Theme Toggle */}
                <div className="border-t p-4 space-y-3">
                    <ThemeToggle />
                    <p className="text-xs text-muted-foreground">
                        Stock Radar v1.0
                    </p>
                </div>
            </div>
        </aside>
    )
}

