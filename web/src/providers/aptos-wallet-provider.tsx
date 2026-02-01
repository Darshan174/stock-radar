"use client"

import { AptosWalletAdapterProvider } from "@aptos-labs/wallet-adapter-react"
import { Network } from "@aptos-labs/ts-sdk"

export function AptosWalletProvider({ children }: { children: React.ReactNode }) {
  return (
    <AptosWalletAdapterProvider
      optInWallets={["Petra"]}
      autoConnect={false}
      dappConfig={{ network: Network.TESTNET }}
    >
      {children}
    </AptosWalletAdapterProvider>
  )
}
