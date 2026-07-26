export type BrowserSearchItem = {
  id: string;
  situation: string;
  document: string;
  vector: number[];
};

export type BrowserSearchIndex = {
  model: string;
  dimensions: number;
  pooling?: "mean" | "cls";
  queryPrefix?: string;
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

async function embedQuery(
  model: string,
  query: string,
  pooling: "mean" | "cls" = "mean",
): Promise<number[]> {
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
    pooling,
    normalize: true,
  });
  return Array.from(output.data);
}

export type TacticalQueryIntent = {
  unsupported: boolean;
  preferred?: "high_press" | "central_screen" | "unstructured";
  suppressed?: "high_press" | "central_screen" | "unstructured";
};

export function tacticalQueryIntent(query: string): TacticalQueryIntent {
  const clean = query.toLowerCase().replaceAll("-", " ");
  if (/\b(corner(?:\s+kick)?|free\s+kick|penalty|red\s+card|throw[\s-]?in|low\s+block)\b/.test(clean)) {
    return { unsupported: true };
  }
  if (
    /\b(not (?:a )?(?:high )?press(?:ing)?|not pressing|without (?:local )?pressure|no (?:local )?pressure|limited coordinated pressure|passive defensive shape|unopposed (?:build[\s-]?up)?|free player in possession|lots of space around the ball|disorganized pressure|no nearby defender|low pressure on the ball)\b/.test(clean)
  ) {
    return {
      unsupported: false,
      preferred: "unstructured",
      suppressed: "high_press",
    };
  }
  if (
    /\b(high press|aggressive press|intense pressure high|counter[\s-]?press|close down|swarm(?:ing)? the ball|rushed pass|pressing the back line|compress(?:ing)? the ball|stop short build[\s-]?up)\b/.test(clean)
  ) {
    return { unsupported: false, preferred: "high_press" };
  }
  if (
    /\b(central screen|block the middle|central passing lanes?|through the cent(?:er|re)|midfield screen|central lanes?|number ten space|central defensive shape)\b/.test(clean)
  ) {
    return { unsupported: false, preferred: "central_screen" };
  }
  return { unsupported: false };
}

export async function hybridSearch(
  query: string,
  index: BrowserSearchIndex,
): Promise<HybridResult[]> {
  const intent = tacticalQueryIntent(query);
  if (intent.unsupported) return [];
  const embeddedQuery = `${index.queryPrefix ?? ""}${query}`;
  const queryVector = await embedQuery(
    index.model,
    embeddedQuery,
    index.pooling ?? "mean",
  );
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
    let score = index.weights.cosine * semantic
      + index.weights.bm25 * lexical;
    if (item.situation === intent.preferred) score += 0.18;
    if (item.situation === intent.suppressed) score -= 0.18;
    return {
      id: item.id,
      score: Math.min(1, Math.max(0, score)),
      cosine: cosine[itemIndex],
      bm25: bm25[itemIndex],
    };
  }).sort((left, right) => right.score - left.score);
}
