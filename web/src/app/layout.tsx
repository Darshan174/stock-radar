import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppSidebar } from "@/components/app-sidebar";
import { MainContent } from "@/components/main-content";
import { AptosWalletProvider } from "@/providers/aptos-wallet-provider";
import { SidebarProvider } from "@/providers/sidebar-provider";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Stock Radar",
  description: "AI-powered stock analysis dashboard",
};

// Inline script to set theme before hydration (prevents flash)
const themeScript = `
(function() {
  try {
    var theme = localStorage.getItem('theme');
    if (theme === 'light') {
      document.documentElement.classList.remove('dark');
    } else if (theme === 'device') {
      if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    } else {
      document.documentElement.classList.add('dark');
    }
  } catch(e) {}
})()
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`dark ${inter.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className={`${inter.className} antialiased bg-background text-foreground`}>
        <AptosWalletProvider>
          <SidebarProvider>
            <AppSidebar />
            <MainContent>
              {children}
            </MainContent>
          </SidebarProvider>
        </AptosWalletProvider>
      </body>
    </html>
  );
}
