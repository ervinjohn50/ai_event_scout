"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import styles from "./Nav.module.css";

const LINKS = [
  { href: "/", label: "Chat" },
  { href: "/shortlist", label: "Shortlist" },
  { href: "/stats", label: "Stats" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className={styles.nav} aria-label="Main">
      {LINKS.map(({ href, label }) => (
        <Link key={href} href={href} aria-current={pathname === href ? "page" : undefined}>
          {label}
        </Link>
      ))}
    </nav>
  );
}
