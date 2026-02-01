import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppSidebar } from "@/components/app-sidebar";
import { AptosWalletProvider } from "@/providers/aptos-wallet-provider";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Stock Radar",
  description: "AI-powered stock analysis dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`dark ${inter.variable}`}>
      <body className={`${inter.className} antialiased bg-background text-foreground`}>
        <AptosWalletProvider>
          <AppSidebar />
          <main className="ml-64 min-h-screen">
            {children}
          </main>
        </AptosWalletProvider>
      </body>
    </html>
  );
}

