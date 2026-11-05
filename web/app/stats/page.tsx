import type { Metadata } from "next";

import { Stats } from "@/components/Stats";

export const metadata: Metadata = { title: "Stats | Event Scout" };

export default function StatsPage() {
  return <Stats />;
}
