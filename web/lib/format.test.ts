import { describe, expect, it } from "vitest";

import { engineLabel, formatDollars, formatEventDate, formatNumber, formatTotalCost, safeUrl } from "./format";

describe("formatEventDate", () => {
  it("keeps the same calendar day, even west of UTC", () => {
    expect(formatEventDate("2026-11-10")).toBe("Tue, Nov 10, 2026");
  });

  it("leaves other text alone", () => {
    expect(formatEventDate("TBA")).toBe("TBA");
  });
});

describe("safeUrl", () => {
  it.each(["https://example.com/t", "http://example.com/t"])("allows %s", (url) => {
    expect(safeUrl(url)).toBe(url);
  });

  it.each(["javascript:alert(1)", "JaVaScRiPt:alert(1)", " javascript:alert(1)", "data:text/html,x", "not a url", "", null])(
    "blocks %j",
    (url) => {
      expect(safeUrl(url)).toBeNull();
    },
  );
});

describe("numbers", () => {
  it("formats averages and shows a dash when there's nothing to average", () => {
    expect(formatNumber(2.34)).toBe("2.3");
    expect(formatNumber(4874.6, 0)).toBe("4,875");
    expect(formatNumber(null)).toBe("–");
    expect(formatNumber(0)).toBe("0");
  });

  it("keeps small dollar amounts visible", () => {
    expect(formatDollars(0)).toBe("$0.00");
    expect(formatDollars(0.004)).toBe("under $0.01");
    expect(formatDollars(1.5)).toBe("$1.50");
  });

  it("doesn't call a total complete when some questions have no known price", () => {
    expect(formatTotalCost(1.5, 0)).toBe("$1.50");
    expect(formatTotalCost(0, 0)).toBe("$0.00");
    expect(formatTotalCost(1.5, 2)).toBe("$1.50 + 2 unpriced");
    expect(formatTotalCost(0, 2)).toBe("Unknown");
  });
});

describe("engineLabel", () => {
  it("names known engines and passes through others", () => {
    expect(engineLabel("custom")).toBe("Custom loop");
    expect(engineLabel("langgraph")).toBe("LangGraph");
    expect(engineLabel("new-engine")).toBe("new-engine");
  });
});
