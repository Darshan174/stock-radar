"use client"

import { useSidebar } from "@/providers/sidebar-provider"
import { FloatingChat } from "@/components/chat-assistant"

export function MainContent({ children }: { children: React.ReactNode }) {
  const { collapsed, hidden } = useSidebar()

  return (
    <main
      className={`${hidden ? "ml-0" : collapsed ? "ml-16" : "ml-64"} min-h-screen transition-[margin] duration-300 ease-out`}
    >
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 -z-10 bg-black dark:bg-black"
      />
      <div className="relative z-10">{children}</div>
      {!hidden && <FloatingChat />}
    </main>
  )
}
