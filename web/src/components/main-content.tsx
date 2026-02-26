"use client"

import { useSidebar } from "@/providers/sidebar-provider"

export function MainContent({ children }: { children: React.ReactNode }) {
  const { collapsed, hidden } = useSidebar()

  return (
    <main
      className={`${hidden ? "ml-0" : collapsed ? "ml-16" : "ml-64"} min-h-screen transition-[margin] duration-200`}
    >
      {children}
    </main>
  )
}
