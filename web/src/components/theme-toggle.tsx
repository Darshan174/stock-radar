"use client"

import { useState, useEffect } from "react"
import { cn } from "@/lib/utils"

type ThemeMode = "dark" | "light" | "system"

const THEME_KEY = "theme"

function normalizeTheme(value: string | null): ThemeMode {
    if (value === "dark" || value === "light" || value === "system") return value
    // Backward compatibility with older "device" key.
    if (value === "device") return "system"
    return "system"
}

function applyTheme(mode: ThemeMode) {
    if (mode === "light") {
        document.documentElement.classList.remove("dark")
    } else if (mode === "dark") {
        document.documentElement.classList.add("dark")
    } else {
        // Follow system preference.
        if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
            document.documentElement.classList.add("dark")
        } else {
            document.documentElement.classList.remove("dark")
        }
    }
}

const modes: { value: ThemeMode; label: string }[] = [
    { value: "dark", label: "Dark" },
    { value: "light", label: "Light" },
    { value: "system", label: "System" },
]

export function ThemeToggle() {
    const [theme, setTheme] = useState<ThemeMode>("system")

    useEffect(() => {
        const savedTheme = normalizeTheme(localStorage.getItem(THEME_KEY))
        if (savedTheme !== theme) {
            setTheme(savedTheme)
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    useEffect(() => {
        applyTheme(theme)
    }, [theme])

    // Listen for system theme changes only in "system" mode.
    useEffect(() => {
        if (theme !== "system") return

        const mq = window.matchMedia("(prefers-color-scheme: dark)")
        const handler = () => applyTheme("system")
        mq.addEventListener("change", handler)
        return () => mq.removeEventListener("change", handler)
    }, [theme])

    const selectTheme = (mode: ThemeMode) => {
        setTheme(mode)
        localStorage.setItem(THEME_KEY, mode)
    }

    return (
        <div className="inline-flex w-full items-center rounded-full border border-sky-200/85 bg-white/85 p-1 backdrop-blur-sm dark:border-border/70 dark:bg-background/45">
            {modes.map((mode) => (
                <button
                    key={mode.value}
                    onClick={() => selectTheme(mode.value)}
                    title={mode.label}
                    className={cn(
                        "h-7 flex-1 rounded-full px-2.5 text-[11px] font-medium tracking-wide transition-all duration-200",
                        theme === mode.value
                            ? "bg-sky-600 text-white shadow-[0_8px_20px_-14px_rgba(37,99,235,0.55)] dark:bg-primary dark:text-primary-foreground dark:shadow-[0_8px_20px_-14px_rgba(34,211,238,0.8)]"
                            : "text-slate-600 hover:bg-sky-100 hover:text-slate-900 dark:text-muted-foreground dark:hover:bg-accent/60 dark:hover:text-foreground"
                    )}
                >
                    {mode.label}
                </button>
            ))}
        </div>
    )
}
