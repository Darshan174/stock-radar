"use client"

import { useState, useEffect } from "react"
import { Moon, Sun, Monitor } from "lucide-react"
import { cn } from "@/lib/utils"

type ThemeMode = "dark" | "light" | "device"

function applyTheme(mode: ThemeMode) {
    if (mode === "light") {
        document.documentElement.classList.remove("dark")
    } else if (mode === "dark") {
        document.documentElement.classList.add("dark")
    } else {
        // Device preference
        if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
            document.documentElement.classList.add("dark")
        } else {
            document.documentElement.classList.remove("dark")
        }
    }
}

const modes: { value: ThemeMode; label: string; icon: typeof Sun }[] = [
    { value: "light", label: "Light", icon: Sun },
    { value: "dark", label: "Dark", icon: Moon },
    { value: "device", label: "Device", icon: Monitor },
]

export function ThemeToggle() {
    const [theme, setTheme] = useState<ThemeMode>("dark")

    useEffect(() => {
        const saved = localStorage.getItem("theme") as ThemeMode | null
        if (saved && ["dark", "light", "device"].includes(saved)) {
            setTheme(saved)
            applyTheme(saved)
        }
    }, [])

    // Listen for system theme changes when in device mode
    useEffect(() => {
        if (theme !== "device") return

        const mq = window.matchMedia("(prefers-color-scheme: dark)")
        const handler = () => applyTheme("device")
        mq.addEventListener("change", handler)
        return () => mq.removeEventListener("change", handler)
    }, [theme])

    const selectTheme = (mode: ThemeMode) => {
        setTheme(mode)
        applyTheme(mode)
        localStorage.setItem("theme", mode)
    }

    return (
        <div className="flex items-center rounded-lg border p-1 gap-0.5">
            {modes.map((mode) => (
                <button
                    key={mode.value}
                    onClick={() => selectTheme(mode.value)}
                    title={mode.label}
                    className={cn(
                        "flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors",
                        theme === mode.value
                            ? "bg-primary text-primary-foreground"
                            : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    )}
                >
                    <mode.icon className="h-3.5 w-3.5" />
                    {mode.label}
                </button>
            ))}
        </div>
    )
}
