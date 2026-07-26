import { describe, expect, it } from "vitest";
import { bm25Scores, tacticalQueryIntent } from "./browser-retrieval";

const documents = [
  "Tactical situation high press dense pressure around build up",
  "Tactical situation central screen central passing lanes screened",
  "Tactical situation no local pressure limited coordinated pressure",
];

describe("browser BM25 retrieval", () => {
  it("ranks central screen first for central press", () => {
    const scores = bm25Scores("central press", documents);
    expect(scores.indexOf(Math.max(...scores))).toBe(1);
  });

  it("ranks high press first for high press", () => {
    const scores = bm25Scores("high press", documents);
    expect(scores.indexOf(Math.max(...scores))).toBe(0);
  });
});

describe("tactical query intent", () => {
  it.each([
    ["not a high press", "unstructured"],
    ["defenders are not pressing", "unstructured"],
    ["passive defensive shape", "unstructured"],
    ["intense pressure high up the pitch", "high_press"],
    ["block the number ten space", "central_screen"],
  ])("maps %s to %s", (query, expected) => {
    expect(tacticalQueryIntent(query).preferred).toBe(expected);
  });

  it.each(["corner kick", "low block", "red card"])(
    "abstains for unsupported query %s",
    (query) => {
      expect(tacticalQueryIntent(query).unsupported).toBe(true);
    },
  );
});
