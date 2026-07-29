import { describe, expect, it } from "vitest";
import { bm25Scores, tacticalQueryIntent } from "./browser-retrieval";

const documents = [
  "Tactical situation high press near the wing and touchline",
  "Tactical situation central high press around build up",
  "Tactical situation compact low block near the penalty area",
];

describe("browser BM25 retrieval", () => {
  it("ranks central high press first for central press", () => {
    const scores = bm25Scores("central build up", documents);
    expect(scores.indexOf(Math.max(...scores))).toBe(1);
  });

  it("ranks wing press first for a wing query", () => {
    const scores = bm25Scores("wing touchline", documents);
    expect(scores.indexOf(Math.max(...scores))).toBe(0);
  });
});

describe("tactical query intent", () => {
  it.each([
    ["high press near the wing", "high_press_wing"],
    ["central high press", "high_press_central"],
    ["middle third press", "medium_press"],
    ["compact low block near the box", "low_block"],
  ])("maps %s to %s", (query, expected) => {
    expect(tacticalQueryIntent(query).preferred).toBe(expected);
  });

  it.each(["corner kick", "free kick", "red card"])(
    "abstains for unsupported query %s",
    (query) => {
      expect(tacticalQueryIntent(query).unsupported).toBe(true);
    },
  );
});
