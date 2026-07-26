# PressLens

![PressLens — search the press and inspect the evidence](figure.png)

PressLens is a football-intelligence application for retrieving tactical
situations from match video with natural-language queries. It combines
browser-side MiniLM embeddings, BM25 text matching, reconstructed pitch graphs,
and synchronized video evidence.

Live application: <https://lqh52.github.io/presslens/>

## The research idea

Coaches, managers, analysts, and players should be able to find a relevant
moment without writing code or learning a database query language. They should
be able to ask for it in everyday football language—for example, “show me a
high press that forces play back inside”—and review the closest match
situations.

Off-the-shelf vision-language models are useful for broad video semantics, but
fine-grained tactical concepts depend on team shape, spacing, pitch location,
pressure around the ball, and movement over time. These distinctions are not
well represented in generic training data, while sufficiently large,
expert-labelled tactical video datasets are expensive to create.

PressLens explores a structure-first alternative. Synthetic player-and-ball
graphs provide controlled examples of tactical patterns. A model can learn
those patterns in normalized pitch space, while game-state reconstruction maps
real broadcast video into the same graph representation. The resulting
tactical class and structural evidence can then support natural-language
retrieval of real video.

The current application is the retrieval and evidence layer of that research
direction. It combines text-embedding similarity with BM25 and presents the
corresponding broadcast and reconstructed tactical views. Learning retrieval
embeddings directly from synthetic and reconstructed graphs is the next
modelling stage.

## What is in this repository?

The repository has two distinct parts:

- **Application:** the Next.js interface, browser retrieval, deployment
  catalogue, and media integration.
- **Research:** scripts for dataset preparation, game-state reconstruction,
  graph construction, weak supervision, model training, review, and evaluation.

Raw SoccerNet video, generated graphs, model weights, local annotations, and
credentials are intentionally excluded from Git.

## Use the application

Requirements:

- Node.js 20 or newer
- npm 10 or newer
- Git

Clone and start the development server:

```bash
git clone https://github.com/lqh52/presslens.git
cd presslens
npm install
cp .env.local.example .env.local
npm run dev
```

Open <http://localhost:3000>.

On Windows PowerShell, replace the copy command with:

```powershell
Copy-Item .env.local.example .env.local
```

The example environment file contains no credentials. The development and
build commands automatically restore the committed retrieval catalogue into
`public/demo`.

The first search downloads a quantized MiniLM model from Hugging Face and
caches it in the browser. Search then runs locally: 65% normalized embedding
cosine similarity and 35% normalized BM25 lexical relevance.

Run the application checks:

```bash
npm test
npm run build
```

## Deploy the application

The hosted version uses:

- GitHub Pages for the static interface and retrieval catalogue;
- object storage for video and image assets;
- Hugging Face for the browser-compatible MiniLM model.

See [DEPLOYMENT.md](DEPLOYMENT.md) for setup and update instructions.

## Reproduce or extend the research

Research setup is separate from normal application use. It covers:

1. creating a Python environment;
2. requesting and downloading SoccerNet data;
3. defining the scientific task and hypotheses;
4. producing calibrated game-state graphs;
5. generating synthetic and weak tactical labels;
6. training graph and team-identity models;
7. conducting blinded review and quantitative evaluation;
8. exporting reviewed examples to the application.

See [RESEARCH.md](RESEARCH.md) for the reproducible workflow and
[RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the tactical class definitions shown
inside the application.

## Supported tactical classes

- **Central screen:** defenders occupy central progression lanes ahead of the
  ball.
- **High press:** several defenders compress the ball area while the possession
  team builds in its defensive third.
- **No local pressure:** coordinated pressure near the ball is absent or
  structurally unclear.

Touchline-trap classes remain experimental and are not presented as supported
classes without retained reviewed examples.

## Data and licensing

Dataset access and usage conditions are described on the
[SoccerNet data page](https://www.soccer-net.org/data). Video access may
require an NDA. Footage should be stored and shared in accordance with those
conditions. Local credentials can be kept in `.env`, which is ignored by Git.
