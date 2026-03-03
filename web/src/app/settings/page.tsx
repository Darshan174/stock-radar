"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Check,
  Loader2,
  Bell,
  Mail,
  SlidersHorizontal,
  Gauge,
  ShieldCheck,
  Server,
  MessageSquare,
  Sparkles,
} from "lucide-react"
import { hasSupabaseEnv } from "@/lib/supabase"

interface Settings {
  slackEnabled: boolean
  slackChannel: string
  emailEnabled: boolean
  email: string
  defaultMode: "intraday" | "longterm"
  confidenceThreshold: number
}

const DEFAULT_SETTINGS: Settings = {
  slackEnabled: true,
  slackChannel: "",
  emailEnabled: false,
  email: "",
  defaultMode: "intraday",
  confidenceThreshold: 70,
}

function loadSettings(): Settings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS
  const savedSettings = localStorage.getItem("stock-radar-settings")
  if (!savedSettings) return DEFAULT_SETTINGS

  try {
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(savedSettings) as Partial<Settings>) }
  } catch {
    return DEFAULT_SETTINGS
  }
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>(loadSettings)
  const [slackDialog, setSlackDialog] = useState(false)
  const [emailDialog, setEmailDialog] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  // Save settings to localStorage
  function saveSettings(newSettings: Settings) {
    setSettings(newSettings)
    localStorage.setItem("stock-radar-settings", JSON.stringify(newSettings))
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  function handleSlackSave() {
    setSaving(true)
    setTimeout(() => {
      saveSettings({ ...settings })
      setSaving(false)
      setSlackDialog(false)
    }, 500)
  }

  function handleEmailSave() {
    setSaving(true)
    setTimeout(() => {
      saveSettings({ ...settings })
      setSaving(false)
      setEmailDialog(false)
    }, 500)
  }

  function setDefaultMode(mode: Settings["defaultMode"]) {
    saveSettings({ ...settings, defaultMode: mode })
  }

  function setThreshold(value: number) {
    saveSettings({ ...settings, confidenceThreshold: value })
  }

  const enabledChannels = Number(settings.slackEnabled) + Number(settings.emailEnabled)
  const thresholds = [50, 60, 70, 80, 90]

  return (
    <div className="app-page">
      <div className="app-page-header">
        <div className="flex items-center gap-3">
          <h1 className="app-page-title">Settings</h1>
          {saved && (
            <span className="inline-flex items-center gap-1 rounded-full border border-emerald-400/35 bg-emerald-400/10 px-2 py-0.5 text-xs font-medium text-emerald-500">
              <Check className="h-4 w-4" /> Saved
            </span>
          )}
        </div>
        <p className="app-page-subtitle">Configure notifications, analysis behavior, and environment readiness.</p>
      </div>

      <div className="mb-4 grid gap-4 sm:grid-cols-3">
        <Card className="border-sky-300/30 bg-gradient-to-br from-sky-500/10 to-indigo-500/5">
          <CardContent className="pt-5">
            <p className="text-xs text-muted-foreground">Default Analysis</p>
            <div className="mt-1 flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-sky-500" />
              <p className="text-lg font-semibold">
                {settings.defaultMode === "intraday" ? "Intraday" : "Long-term"}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card className="border-teal-300/30 bg-gradient-to-br from-teal-500/10 to-emerald-500/5">
          <CardContent className="pt-5">
            <p className="text-xs text-muted-foreground">Confidence Threshold</p>
            <div className="mt-1 flex items-center gap-2">
              <Gauge className="h-4 w-4 text-teal-500" />
              <p className="text-lg font-semibold">{settings.confidenceThreshold}%</p>
            </div>
          </CardContent>
        </Card>
        <Card className="border-violet-300/30 bg-gradient-to-br from-violet-500/10 to-fuchsia-500/5">
          <CardContent className="pt-5">
            <p className="text-xs text-muted-foreground">Notification Channels</p>
            <div className="mt-1 flex items-center gap-2">
              <Bell className="h-4 w-4 text-violet-500" />
              <p className="text-lg font-semibold">{enabledChannels} Enabled</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid max-w-5xl gap-4 lg:grid-cols-2">
        <Card className="border-cyan-300/25">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-4 w-4 text-cyan-500" />
              Notifications
            </CardTitle>
            <CardDescription>Choose where alerts and summaries are delivered.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between rounded-xl border p-3">
              <div className="flex items-center gap-4">
                <Switch
                  checked={settings.slackEnabled}
                  onCheckedChange={(checked) => saveSettings({ ...settings, slackEnabled: checked })}
                />
                <div>
                  <p className="font-medium flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-cyan-500" />
                    Slack Notifications
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {settings.slackEnabled ? "Enabled" : "Disabled"}
                  </p>
                </div>
              </div>
              <Button variant="outline" size="sm" onClick={() => setSlackDialog(true)}>
                Configure
              </Button>
            </div>
            <div className="flex items-center justify-between rounded-xl border p-3">
              <div className="flex items-center gap-4">
                <Switch
                  checked={settings.emailEnabled}
                  onCheckedChange={(checked) => saveSettings({ ...settings, emailEnabled: checked })}
                />
                <div>
                  <p className="font-medium flex items-center gap-2">
                    <Mail className="h-4 w-4 text-indigo-500" />
                    Email Notifications
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {settings.emailEnabled ? settings.email || "Not configured" : "Disabled"}
                  </p>
                </div>
              </div>
              <Button variant="outline" size="sm" onClick={() => setEmailDialog(true)}>
                Configure
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="border-emerald-300/25">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-emerald-500" />
              Analysis Settings
            </CardTitle>
            <CardDescription>Tune how strict and how often analysis signals are triggered.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-xl border p-3">
              <div>
                <p className="font-medium">Default Mode</p>
                <p className="text-sm text-muted-foreground">
                  {settings.defaultMode === "intraday" ? "Short-term trading" : "Long-term investing"}
                </p>
              </div>
              <div className="mt-3 inline-flex w-full rounded-full border p-1 sm:w-auto">
                <Button
                  variant={settings.defaultMode === "intraday" ? "default" : "ghost"}
                  size="sm"
                  className="rounded-full"
                  onClick={() => setDefaultMode("intraday")}
                >
                  Intraday
                </Button>
                <Button
                  variant={settings.defaultMode === "longterm" ? "default" : "ghost"}
                  size="sm"
                  className="rounded-full"
                  onClick={() => setDefaultMode("longterm")}
                >
                  Long-term
                </Button>
              </div>
            </div>
            <div className="rounded-xl border p-3">
              <div className="flex items-center justify-between">
                <p className="font-medium">Confidence Threshold</p>
                <p className="text-sm font-semibold text-emerald-500">{settings.confidenceThreshold}%</p>
              </div>
              <p className="mt-1 text-sm text-muted-foreground">Minimum confidence for alerts</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {thresholds.map((threshold) => (
                  <Button
                    key={threshold}
                    variant={settings.confidenceThreshold === threshold ? "default" : "outline"}
                    size="sm"
                    className="h-7 px-2.5 text-xs"
                    onClick={() => setThreshold(threshold)}
                  >
                    {threshold}%
                  </Button>
                ))}
              </div>
              <div className="mt-3 h-1.5 rounded-full bg-muted">
                <div
                  className="h-1.5 rounded-full bg-emerald-500 transition-all duration-300"
                  style={{ width: `${settings.confidenceThreshold}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-violet-300/25 lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-4 w-4 text-violet-500" />
              Environment Status
            </CardTitle>
            <CardDescription>Quick readiness checks for local and deployed setup.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 text-sm sm:grid-cols-2">
            <div className="flex items-center justify-between rounded-lg border p-3">
              <span className="text-muted-foreground">Supabase Client Env</span>
              <span className={hasSupabaseEnv ? "text-green-500 flex items-center gap-1" : "text-amber-500 flex items-center gap-1"}>
                <ShieldCheck className="h-3.5 w-3.5" />
                {hasSupabaseEnv ? "Configured" : "Missing"}
              </span>
            </div>
            <div className="flex items-center justify-between rounded-lg border p-3">
              <span className="text-muted-foreground">Slack Channel</span>
              <span className={settings.slackEnabled ? "text-green-500" : "text-muted-foreground"}>
                {settings.slackEnabled ? "Enabled" : "Disabled"}
              </span>
            </div>
            <div className="flex items-center justify-between rounded-lg border p-3">
              <span className="text-muted-foreground">Email Channel</span>
              <span className={settings.emailEnabled ? "text-green-500" : "text-muted-foreground"}>
                {settings.emailEnabled ? "Enabled" : "Disabled"}
              </span>
            </div>
            <div className="flex items-center justify-between rounded-lg border p-3">
              <span className="text-muted-foreground">AI Mode</span>
              <span className="text-cyan-500">{settings.defaultMode === "intraday" ? "Intraday" : "Long-term"}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Slack Configuration Dialog */}
      <Dialog open={slackDialog} onOpenChange={setSlackDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Slack Configuration</DialogTitle>
            <DialogDescription>
              Slack notifications are configured via environment variables.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Channel ID</label>
              <Input
                value={settings.slackChannel}
                onChange={(e) => setSettings({ ...settings, slackChannel: e.target.value })}
                placeholder="C0A8TPLMJ92"
              />
              <p className="text-xs text-muted-foreground">
                Set SLACK_CHANNEL_ID in your .env file
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setSlackDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleSlackSave} disabled={saving}>
              {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Email Configuration Dialog */}
      <Dialog open={emailDialog} onOpenChange={setEmailDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Email Notifications</DialogTitle>
            <DialogDescription>
              Configure email for daily summaries (coming soon)
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Email Address</label>
              <Input
                type="email"
                value={settings.email}
                onChange={(e) => setSettings({ ...settings, email: e.target.value })}
                placeholder="you@example.com"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEmailDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleEmailSave} disabled={saving}>
              {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
