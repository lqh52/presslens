export type BrowserSearchItem = {
  id: string;
  situation: string;
  document: string;
  vector: number[];
};

export type BrowserSearchIndex = {
  model: string;
  dimensions: number;
  weights: { cosine: number; bm25: number };
  items: BrowserSearchItem[];
};

export type HybridResult = {
  id: string;
  score: number;
  cosine: number;
  bm25: number;
};

const tokens = (text: string) =>
  text.toLowerCase().replaceAll("-", " ").match(/[a-z0-9]+/g) ?? [];

export function bm25Scores(query: string, documents: string[]): number[] {
  const corpus = documents.map(tokens);
  const terms = [...new Set(tokens(query))];
  const averageLength = corpus.reduce((sum, row) => sum + row.length, 0)
    / Math.max(corpus.length, 1);
  const scores = documents.map(() => 0);
  for (const term of terms) {
    const frequencies = corpus.map(
      (row) => row.filter((token) => token === term).length,
    );
    const documentFrequency = frequencies.filter(Boolean).length;
    if (!documentFrequency) continue;
    const inverseDocumentFrequency = Math.log(
      1 + (corpus.length - documentFrequency + 0.5)
        / (documentFrequency + 0.5),
    );
    frequencies.forEach((frequency, index) => {
      const denominator = frequency + 1.5 * (
        0.25 + 0.75 * corpus[index].length / Math.max(averageLength, 1)
      );
      scores[index] += inverseDocumentFrequency
        * frequency * 2.5 / Math.max(denominator, 1e-8);
    });
  }
  return scores;
}

const dot = (left: number[], right: number[]) =>
  left.reduce((total, value, index) => total + value * right[index], 0);

let extractorPromise: Promise<
  (text: string, options: object) => Promise<{ data: Float32Array }>
> | null = null;

async function embedQuery(model: string, query: string): Promise<number[]> {
  if (!extractorPromise) {
    extractorPromise = import("@huggingface/transformers").then(
      async ({ pipeline }) => pipeline(
        "feature-extraction",
        model,
        { dtype: "q8" },
      ) as unknown as (
        text: string,
        options: object,
      ) => Promise<{ data: Float32Array }>,
    );
  }
  const extractor = await extractorPromise;
  const output = await extractor(query, {
    pooling: "mean",
    normalize: true,
  });
  return Array.from(output.data);
}

export async function hybridSearch(
  query: string,
  index: BrowserSearchIndex,
): Promise<HybridResult[]> {
  const queryVector = await embedQuery(index.model, query);
  if (queryVector.length !== index.dimensions) {
    throw new Error(
      `Embedding dimension ${queryVector.length} does not match index ${index.dimensions}`,
    );
  }
  const cosine = index.items.map((item) => dot(queryVector, item.vector));
  const bm25 = bm25Scores(
    query,
    index.items.map((item) => item.document),
  );
  const lexicalMaximum = Math.max(...bm25, 0);
  return index.items.map((item, itemIndex) => {
    const semantic = Math.min(
      1,
      Math.max(0, (cosine[itemIndex] + 1) / 2),
    );
    const lexical = lexicalMaximum > 0
      ? bm25[itemIndex] / lexicalMaximum
      : 0;
    return {
      id: item.id,
      score: index.weights.cosine * semantic
        + index.weights.bm25 * lexical,
      cosine: cosine[itemIndex],
      bm25: bm25[itemIndex],
    };
  }).sort((left, right) => right.score - left.score);
}
