"use client"

import { useState, useEffect } from "react"
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
import { Check, Loader2 } from "lucide-react"

interface Settings {
  slackEnabled: boolean
  slackChannel: string
  emailEnabled: boolean
  email: string
  defaultMode: "intraday" | "longterm"
  confidenceThreshold: number
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>({
    slackEnabled: true,
    slackChannel: "",
    emailEnabled: false,
    email: "",
    defaultMode: "intraday",
    confidenceThreshold: 70,
  })
  const [slackDialog, setSlackDialog] = useState(false)
  const [emailDialog, setEmailDialog] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  // Load settings from localStorage
  useEffect(() => {
    const savedSettings = localStorage.getItem("stock-radar-settings")
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings))
    }
  }, [])

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

  function toggleMode() {
    const newMode = settings.defaultMode === "intraday" ? "longterm" : "intraday"
    saveSettings({ ...settings, defaultMode: newMode })
  }

  function cycleThreshold() {
    const thresholds = [50, 60, 70, 80, 90]
    const currentIndex = thresholds.indexOf(settings.confidenceThreshold)
    const nextIndex = (currentIndex + 1) % thresholds.length
    saveSettings({ ...settings, confidenceThreshold: thresholds[nextIndex] })
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <div className="flex items-center gap-2">
          <h1 className="text-3xl font-bold">Settings</h1>
          {saved && (
            <span className="text-sm text-green-500 flex items-center gap-1">
              <Check className="h-4 w-4" /> Saved
            </span>
          )}
        </div>
        <p className="text-muted-foreground">Configure your preferences</p>
      </div>

      <div className="grid gap-4 max-w-2xl">
        <Card>
          <CardHeader>
            <CardTitle>Notifications</CardTitle>
            <CardDescription>Configure how you receive alerts</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Switch
                  checked={settings.slackEnabled}
                  onCheckedChange={(checked) => saveSettings({ ...settings, slackEnabled: checked })}
                />
                <div>
                  <p className="font-medium">Slack Notifications</p>
                  <p className="text-sm text-muted-foreground">
                    {settings.slackEnabled ? "Enabled" : "Disabled"}
                  </p>
                </div>
              </div>
              <Button variant="outline" size="sm" onClick={() => setSlackDialog(true)}>
                Configure
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Switch
                  checked={settings.emailEnabled}
                  onCheckedChange={(checked) => saveSettings({ ...settings, emailEnabled: checked })}
                />
                <div>
                  <p className="font-medium">Email Notifications</p>
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

        <Card>
          <CardHeader>
            <CardTitle>Analysis Settings</CardTitle>
            <CardDescription>Customize AI analysis parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Default Mode</p>
                <p className="text-sm text-muted-foreground">
                  {settings.defaultMode === "intraday" ? "Short-term trading" : "Long-term investing"}
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={toggleMode}>
                {settings.defaultMode === "intraday" ? "Intraday" : "Long-term"}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Confidence Threshold</p>
                <p className="text-sm text-muted-foreground">Minimum confidence for alerts</p>
              </div>
              <Button variant="outline" size="sm" onClick={cycleThreshold}>
                {settings.confidenceThreshold}%
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Environment</CardTitle>
            <CardDescription>Connection status</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Supabase</span>
              <span className="text-green-500">Connected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Groq API</span>
              <span className="text-green-500">Configured</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Slack</span>
              <span className={settings.slackEnabled ? "text-green-500" : "text-muted-foreground"}>
                {settings.slackEnabled ? "Enabled" : "Disabled"}
              </span>
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
