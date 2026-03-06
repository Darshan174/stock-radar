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
    CreditCard,
} from "lucide-react"
import { cn } from "@/lib/utils"
// Note: we still use "useSidebar" to check if the intro is hiding the nav
import { useSidebar } from "@/providers/sidebar-provider"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

const ThemeToggle = dynamic(
    () => import("@/components/theme-toggle").then((m) => m.ThemeToggle),
    { ssr: false }
)

const navItems = [
    { href: "/", label: "Dashboard", description: "Market overview and recent signals", icon: LayoutDashboard },
    { href: "/stocks", label: "Stocks", description: "Detailed stock analysis and charts", icon: TrendingUp },
    { href: "/signals", label: "Signals", description: "Trading signals and alerts", icon: Zap },
    { href: "/usage", label: "API Usage", description: "Monitor your API requests and limits", icon: BarChart3 },
    { href: "/x402-demo", label: "x402 Demo", description: "Aptos blockchain integration demo", icon: CreditCard },
    { href: "/settings", label: "Settings", description: "Application preferences and configurations", icon: Settings },
]

export function AppHeader() {
    const pathname = usePathname()
    const { hidden } = useSidebar()

    if (hidden) return null

    return (
        <header
            className="fixed left-0 top-0 z-40 w-full border-b border-border bg-background/80 backdrop-blur-xl"
        >
            <div className="flex h-16 items-center px-4 md:px-6 w-full gap-4 md:gap-8 max-w-7xl mx-auto">
                {/* Logo */}
                <Link href="/" className="flex items-center gap-2 shrink-0">
                    <div className="grid h-8 w-8 place-items-center rounded-md bg-gradient-to-br from-green-500 to-green-400 text-primary-foreground shadow-[0_0_20px_-10px_rgba(34,197,94,0.9)]">
                        <Radar className="h-5 w-5 pointer-events-none" />
                    </div>
                    <span className="text-xl font-semibold whitespace-nowrap text-foreground hidden md:block">
                        Stock Radar
                    </span>
                </Link>

                {/* Navigation */}
                <nav className="flex-1 flex items-center justify-center gap-1 overflow-x-auto no-scrollbar">
                    <TooltipProvider delayDuration={0}>
                        {navItems.map((item) => {
                            const isActive = pathname === item.href ||
                                (item.href !== "/" && pathname.startsWith(item.href))

                            return (
                                <Tooltip key={item.href}>
                                    <TooltipTrigger asChild>
                                        <Link
                                            href={item.href}
                                            className={cn(
                                                "flex items-center gap-2 rounded-full px-3 py-2 text-sm font-medium transition-all duration-200 whitespace-nowrap",
                                                isActive
                                                    ? "bg-green-500/15 text-green-600 dark:text-green-400 shadow-[0_4px_12px_-4px_rgba(34,197,94,0.3)]"
                                                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                                            )}
                                        >
                                            <item.icon className="h-4 w-4 shrink-0" />
                                            <span className="hidden sm:block">{item.label}</span>
                                        </Link>
                                    </TooltipTrigger>
                                    <TooltipContent side="bottom" sideOffset={12} className="bg-popover border-border text-popover-foreground">
                                        <div className="flex flex-col gap-1">
                                            <span className="font-semibold text-green-600 dark:text-green-400">{item.label}</span>
                                            <span className="text-xs text-muted-foreground">{item.description}</span>
                                        </div>
                                    </TooltipContent>
                                </Tooltip>
                            )
                        })}
                    </TooltipProvider>
                </nav>

                {/* Theme Toggle */}
                <div className="flex items-center justify-end shrink-0 gap-4">
                    <ThemeToggle />
                </div>
            </div>
        </header>
    )
}
