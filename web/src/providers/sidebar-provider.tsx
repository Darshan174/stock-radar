"use client"

import { createContext, useContext, useState, ReactNode } from "react"

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
