"use client"

import { useSidebar } from "@/providers/sidebar-provider"

export function MainContent({ children }: { children: React.ReactNode }) {
  const { collapsed, hidden } = useSidebar()

  return (
    <main
      className={`${hidden ? "ml-0" : collapsed ? "ml-16" : "ml-64"} min-h-screen transition-[margin] duration-300 ease-out`}
    >
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_8%_12%,rgba(56,189,248,0.13),transparent_35%),radial-gradient(circle_at_86%_8%,rgba(99,102,241,0.09),transparent_34%),radial-gradient(circle_at_52%_92%,rgba(16,185,129,0.07),transparent_40%)] dark:bg-[radial-gradient(circle_at_8%_12%,rgba(84,240,255,0.14),transparent_34%),radial-gradient(circle_at_86%_8%,rgba(184,152,255,0.12),transparent_33%),radial-gradient(circle_at_52%_92%,rgba(141,255,108,0.08),transparent_40%)]"
      />
      <div className="relative z-10">{children}</div>
    </main>
  )
}
