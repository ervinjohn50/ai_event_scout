import type { NextConfig } from "next";

const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  devIndicators: false, // the floating badge covers the message box on small screens
  experimental: {
    // The forwarding below gives up after 30 seconds by default, but an agent can take
    // longer than that on a question with several tool calls.
    proxyTimeout: 5 * 60 * 1000,
  },
  async rewrites() {
    return [
      {
        // The browser talks only to this server, which forwards API requests. The API
        // keys stay in the Python process, and CORS isn't needed.
        source: "/api/:path*",
        destination: `${API_URL}/api/:path*`,
        // FastAPI only sees this server's address, so this repeats its host name check.
        // Otherwise a malicious site pointing its domain at this computer could get
        // requests forwarded (DNS rebinding).
        has: [{ type: "host", value: "(localhost|127\\.0\\.0\\.1)" }],
      },
    ];
  },
};

export default nextConfig;
