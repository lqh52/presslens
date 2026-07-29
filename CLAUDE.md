# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PressLens is a football-intelligence research prototype for natural-language retrieval of tactical situations from match video. It combines browser-side text embeddings (Snowflake Arctic), BM25 lexical matching, reconstructed pitch graphs, and synchronized video evidence.

The codebase has two distinct parts:
- **Application**: Next.js interface, browser retrieval, deployment catalogue, and media integration
- **Research**: Python scripts for dataset preparation, game-state reconstruction, graph construction, model training, and evaluation

This document focuses on application development. For research workflows, see [RESEARCH.md](RESEARCH.md).

## Quick Start

```powershell
# Install dependencies
npm install

# Copy example environment
Copy-Item .env.local.example .env.local

# Start development server
npm run dev
```

Open http://localhost:3000. On first search, the browser downloads the quantized Arctic embedding model (~500MB) and caches it locally.

## Development Commands

| Task | Command |
|------|---------|
| Start dev server with hot reload | `npm run dev` |
| Build for production | `npm run build` |
| Run all tests (Vitest) | `npm test` |
| Run single test file | `npm test -- lib/browser-retrieval.test.ts` |
| Lint with ESLint | `npm run lint` |
| Build search index (Python) | `npm run build:search-index` |
| Prepare deployment assets | `npm run prepare:deploy` |
| Upload media to Supabase | `npm run upload:supabase` |

## Application Architecture

### Core Data Model

The application centers on three core types defined in `lib/types.ts`:

1. **Clip**: A tactical situation clip with metadata
   - Video references: `video`, `canonicalImage`, `canonicalVideo` (URLs resolved via env config)
   - Tactical metadata: `situation` (high_press, central_screen, etc.), `description`, `evidence`
   - Structural data: `players[]`, `ball`, `match`, `minute`, `timeSeconds`
   - Confidence scores: `confidence`, `ballConfidence`, `visibleNodes`
   - Review status: `reviewDecision`, `labelSource` (expert_review vs graph_classifier)

2. **Player**: Individual player position and role
   - Position: `x`, `y` (normalized pitch coordinates)
   - Velocity: `dx`, `dy`
   - Metadata: `team` (press|build), `role` (player|goalkeeper), `controlsBall`

3. **DemoManifest**: Dataset metadata and clip index
   - Videos: `Array<{id, half, startSeconds, path}>`
   - Clips: `Clip[]`
   - Metadata: `name`, `count`, `reviewStatus`

### Retrieval Pipeline

Search runs entirely in the browser and combines two signals:

1. **Vector similarity (90% weight)**: Arctic text embedding model (`Snowflake/snowflake-arctic-embed-s`) encodes the query and documents. Cosine similarity ranks results.
   - Implementation: `lib/retrieval.ts` contains `cosineSimilarity()` and `rankVectors()`
   - The model is downloaded on first search and cached in IndexedDB

2. **BM25 lexical matching (10% weight)**: Traditional full-text relevance ranking
   - Implementation: `lib/browser-retrieval.ts` contains `hybridSearch()`
   - Combines both scores: `0.9 * cosine + 0.1 * bm25`

Narrow intent guards filter for unsupported situations (set pieces, low blocks, etc.) and abstain when confidence is too low.

### UI Architecture

`components/press-lens.tsx` is the main single-page component:

- **State management**: React hooks track query, results, selected clip, filters, and sync state with browser `sessionStorage`
- **Video synchronization**: Separate refs for broadcast video and canonical pitch video; playback is synchronized when a result is selected
- **Filtering**: Users can narrow results by `situation` type and `reliablePossession` confidence
- **Tabs**: Toggle between "retrieval" results list and "evidence" (player positions, ball location, pitch visualization)

The pitch visualization component (`components/pitch.tsx`) renders player positions and ball on a normalized pitch.

### Static Asset Loading

On component mount, two JSON files are fetched from `/demo/`:

1. `manifest.json`: Metadata about all clips in the dataset
2. `search-index.json`: Pre-computed vectors and metadata for retrieval

Asset URLs are rewritten based on environment configuration:
- `NEXT_PUBLIC_BASE_PATH`: For deployed sites at non-root paths (e.g., GitHub Pages)
- `NEXT_PUBLIC_MEDIA_ASSET_BASE_URL`: For hosting video/images on external CDN (e.g., Supabase)

If no external URL is configured, all media is served from `/demo/` alongside the app.

## Environment Configuration

| Variable | Purpose | Example |
|----------|---------|---------|
| `NEXT_PUBLIC_BASE_PATH` | Site base path for GitHub Pages deployments | `/presslens/` |
| `NEXT_PUBLIC_MEDIA_ASSET_BASE_URL` | External CDN for video and images | `https://project.supabase.co/storage/v1/object/public/media` |

Create `.env.local` from `.env.local.example` (no credentials needed for local development).

## Key Files

- `app/page.tsx` – Entry point; renders PressLens component
- `components/press-lens.tsx` – Main UI component with search, filtering, and video player
- `components/pitch.tsx` – Pitch visualization with player positions and ball
- `lib/types.ts` – Data types: Clip, Player, Situation, DemoManifest
- `lib/retrieval.ts` – Pure vector operations (cosine similarity, ranking)
- `lib/browser-retrieval.ts` – Hybrid search combining embeddings and BM25
- `lib/*.test.ts` – Vitest tests for retrieval logic
- `next.config.ts` – Next.js configuration
- `tsconfig.json` – TypeScript configuration with `@/*` path alias

## Supported Tactical Classes

- **high_press**: Multiple defenders compress the ball area while the possession team builds in its defensive third
- **central_screen**: Defenders occupy central progression lanes ahead of the ball
- **no_local_pressure** (unstructured): Coordinated pressure near the ball is absent or structurally unclear
- **trap_left / trap_right**: Experimental, not included in published catalogue

## Testing

Tests use Vitest and Testing Library:

- `lib/browser-retrieval.test.ts` – Hybrid search logic and BM25/cosine weighting
- `lib/retrieval.test.ts` – Pure vector operations

Run tests with `npm test` or watch mode with `npm test -- --watch`.

## Important Notes

- **Research prototype**: This is a curated, reviewed dataset with 8 clips. It demonstrates the end-to-end workflow and provides inspectable evidence, not full match coverage.
- **Browser-only search**: All retrieval runs in the visitor's browser. No server-side indexing.
- **Data licensing**: Video clips originate from NDA-controlled SoccerNet access. Before deploying with different media, ensure you have rights to share and host the content.
- **Pre-computed assets**: The search index is pre-built offline (via `build_search_index.py`) and bundled as static JSON. Vectors are frozen; adding new clips requires rebuilding the index.

## Deployment

For deploying to GitHub Pages + Supabase, see [DEPLOYMENT.md](DEPLOYMENT.md). Key steps:

1. Prepare reviewed demo data: `python scripts/build_reviewed_web_demo.py`
2. Build search index: `npm run build:search-index`
3. Publish catalogue: `node scripts/sync_deploy_catalog.mjs publish`
4. Upload media to Supabase: `npm run upload:supabase`
