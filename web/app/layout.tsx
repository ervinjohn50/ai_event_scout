import type { Metadata } from "next";

import { ChatProvider } from "@/components/ChatProvider";
import { Nav } from "@/components/Nav";

import "./globals.css";
import styles from "./layout.module.css";

export const metadata: Metadata = {
  title: "Event Scout",
  description: "Chat with an AI agent that finds concerts and live events.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className={styles.shell}>
          <header className={styles.header}>
            <h1 className={styles.title}>Event Scout</h1>
            <Nav />
          </header>
          <main className={styles.main}>
            <ChatProvider>{children}</ChatProvider>
          </main>
        </div>
      </body>
    </html>
  );
}
