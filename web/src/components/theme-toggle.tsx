"use client"

import { useState, useEffect } from "react"
import { Moon, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"

export function ThemeToggle() {
    const [theme, setTheme] = useState<"dark" | "light">("dark")

    useEffect(() => {
        // Check initial theme from document
        const isDark = document.documentElement.classList.contains("dark")
        setTheme(isDark ? "dark" : "light")
    }, [])

    const toggleTheme = () => {
        const newTheme = theme === "dark" ? "light" : "dark"
        setTheme(newTheme)

        if (newTheme === "dark") {
            document.documentElement.classList.add("dark")
        } else {
            document.documentElement.classList.remove("dark")
        }

        // Save preference
        localStorage.setItem("theme", newTheme)
    }

    return (
        <Button
            variant="outline"
            size="sm"
            onClick={toggleTheme}
            className="gap-2"
        >
            {theme === "dark" ? (
                <>
                    <Sun className="h-4 w-4" />
                    Light Mode
                </>
            ) : (
                <>
                    <Moon className="h-4 w-4" />
                    Dark Mode
                </>
            )}
        </Button>
    )
}
