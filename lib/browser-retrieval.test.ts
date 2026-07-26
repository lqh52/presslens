import { describe, expect, it } from "vitest";
import { bm25Scores } from "./browser-retrieval";

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
