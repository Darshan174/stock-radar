"use client"

import { useSidebar } from "@/providers/sidebar-provider"

export function MainContent({ children }: { children: React.ReactNode }) {
  const { hidden } = useSidebar()

  return (
    <main
      className={`${hidden ? "pt-0" : "pt-16"} min-h-screen transition-[padding] duration-300 ease-out`}
    >
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 -z-10 bg-background"
      />
      <div className="relative z-10">{children}</div>
    </main>
  )
}
