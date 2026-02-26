"use client"

import { createContext, useContext, useState, useEffect, ReactNode } from "react"

interface SidebarContextType {
  collapsed: boolean
  hidden: boolean
  toggle: () => void
  setHidden: (hidden: boolean) => void
}

const SidebarContext = createContext<SidebarContextType>({
  collapsed: false,
  hidden: false,
  toggle: () => { },
  setHidden: () => { },
})

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false)
  const [hidden, setHidden] = useState(false)

  useEffect(() => {
    const saved = localStorage.getItem("sidebar-collapsed")
    if (saved === "true") setCollapsed(true)
  }, [])

  const toggle = () => {
    setCollapsed((prev) => {
      const next = !prev
      localStorage.setItem("sidebar-collapsed", String(next))
      return next
    })
  }

  return (
    <SidebarContext.Provider value={{ collapsed, hidden, toggle, setHidden }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebar() {
  return useContext(SidebarContext)
}
