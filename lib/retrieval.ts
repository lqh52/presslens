/** Pure vector operations mirroring the local Python retrieval service. */
export function cosineSimilarity(left: number[], right: number[]): number {
  if (left.length !== right.length || !left.length) {
    throw new Error("Vectors must have the same non-zero dimensionality");
  }
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] ** 2;
    rightNorm += right[index] ** 2;
  }
  if (!leftNorm || !rightNorm) throw new Error("Cosine similarity is undefined for zero vectors");
  return dot / Math.sqrt(leftNorm * rightNorm);
}

export function rankVectors(
  query: number[],
  documents: Array<{ id: string; vector: number[] }>,
): Array<{ id: string; score: number }> {
  return documents
    .map((document) => ({ id: document.id, score: cosineSimilarity(query, document.vector) }))
    .sort((left, right) => right.score - left.score);
}
