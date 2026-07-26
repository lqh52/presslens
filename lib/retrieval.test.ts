import { describe, expect, it } from "vitest";
import { cosineSimilarity, rankVectors } from "./retrieval";

describe("cosine vector retrieval", () => {
  it("returns one for identical directions", () => {
    expect(cosineSimilarity([1, 2, 3], [2, 4, 6])).toBeCloseTo(1);
  });

  it("ranks by vector direction rather than magnitude", () => {
    const results = rankVectors([1, 0], [
      { id: "opposite", vector: [-10, 0] },
      { id: "close", vector: [2, 0.2] },
      { id: "orthogonal", vector: [0, 100] },
    ]);
    expect(results.map((row) => row.id)).toEqual(["close", "orthogonal", "opposite"]);
  });

  it("rejects invalid vectors", () => {
    expect(() => cosineSimilarity([1], [1, 2])).toThrow();
    expect(() => cosineSimilarity([0, 0], [1, 0])).toThrow();
  });
});
