import type { Metadata } from "next";

import { Shortlist } from "@/components/Shortlist";

export const metadata: Metadata = { title: "Shortlist | Event Scout" };

export default function ShortlistPage() {
  return <Shortlist />;
}
