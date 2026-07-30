# PressLens research

## Objective

PressLens studies natural-language retrieval of football pressing situations
from broadcast video. The system reconstructs player and ball state in
canonical pitch coordinates, classifies short tactical sequences with a
spatiotemporal graph model, and presents synchronized broadcast and graph
evidence for review.

The current research question is whether normalized player-and-ball structure
can provide useful supervision for tactical concepts that are difficult to
learn reliably from pixels alone.

## Current system

The current pipeline has five stages:

1. **Detection and tracking.** YOLO26 detects people and the ball. Person
   detections above the retained confidence threshold are associated into
   tracks.
2. **Fixture-local identity labeling.** Representative frames, expanded body
   crops, and track boxes are classified with Gemini 3.6 Flash using reviewed
   examples from the same fixture.
3. **Conservative identity recovery.** Unresolved tracks can be proposed only
   when DINO appearance and torso-colour distances independently agree.
   Nearest-`other` evidence and canonical off-pitch evidence act as vetoes.
   Rejected tracks remain unresolved.
4. **Canonical reconstruction.** Broadcast detections and ball observations
   are projected onto a 105 × 68 metre pitch. Repaired calibration is used
   before graph conversion and tactical inference.
5. **Tactical inference.** A spatiotemporal graph network consumes canonical
   player-and-ball geometry over time. The tactical classifier does not use
   broadcast pixels or visual embeddings.

The graph representation includes team identity, canonical position, short
motion estimates, possession state, goalkeeper role where supported, and the
ball. Same-team structure edges and short cross-team pressure relationships
are shown as inspectable evidence.

## Tactical classes

The current model and product use four classes:

- **High press — wing:** the pressing team engages high with pressure
  concentrated near a flank.
- **High press — central:** the pressing team engages high through central
  build-up lanes.
- **Medium press:** the defending shape engages around the middle third.
- **Low block:** the defending team remains compact near its own penalty area,
  with most of its shape behind the ball.

These labels describe the dominant structure in a short sequence. They are not
event annotations for a single tackle or pressure action.

## Identity supervision

Identity labels are fixture-local because team kits, goalkeepers, officials,
staff, and broadcast conditions vary by match. A small reviewed seed set
contains examples of both teams and the `other` class. These examples are used
as in-context identity references and to validate the fixture model.

Gemini labels tracks from representative frames, expanded body crops, and
track boxes. A team prediction can become a fixture anchor only when it is not
an abstention, the kit is visible, and the track has at least three retained
detections. Visible, non-abstained `other` predictions are retained only as
negative references.

## Tactical supervision

Supervised tactical examples are constructed from synchronized SkillCorner
tracking, dynamic-event, and phase-of-play data:

1. Select pressing-chain events and settled low-block intervals from the
   provider labels.
2. Normalize each sequence so the team in possession attacks from left to
   right on a 105 × 68 metre pitch.
3. Retain only frames with a detected ball and at least eight detected players
   that can be mapped to a team. Extrapolated players are not added.
4. Sample five ordered frames across the labeled interval.
5. Convert players, the ball, motion, possession, and team relations into the
   same graph schema used for projected broadcast video.

The four training labels are defined programmatically before model training:

- **High press — wing:** a pressing-chain event in a source `high_block`
  phase, starting in a wide channel, with a pressing-chain length of at least
  two.
- **High press — central:** a pressing-chain event in a source `high_block`
  phase that does not satisfy the wing rule.
- **Medium press:** a pressing-chain event in a source `medium_block` phase.
- **Low block:** a source `low_block` phase lasting at least four seconds; the
  team out of possession is assigned as the defending team.

Training and validation are separated by fixture so sequences from one match
do not occur in both partitions. Reviewed broadcast-video sequences can be
added as domain supervision after canonical reconstruction. Their labels
belong to the exact temporal window, since different windows from the same
video can represent different tactical phases.

## Conservative identity recovery

Recovery is restricted to unreviewed tracks that are missing an identity
prediction or were labeled `unknown`/abstained. It uses a separate
fixture-local model and follows these rules:

- Each team must have at least three confident anchors. The team prototype is
  the medoid of its anchors.
- DINO appearance and normalized torso-colour distance must independently
  select the same team.
- Every reviewed team seed must be assigned to its expected team with
  DINO/colour agreement.
- Every reviewed `other` seed must be rejected by signal disagreement or by
  being at least as close to an `other` reference as to the proposed team in
  either DINO or colour space.
- The candidate must fall inside the fixture-calibrated team radius and exceed
  the fixture-calibrated separation margin.
- Canonical evidence must contain at least three projected frames, with at
  least 70% of projected positions on the pitch. Off-pitch evidence is a hard
  team veto.

The fixture model is disabled if its seed checks fail. A candidate that fails
any gate remains `unknown` and requires review. Recovery never automatically
assigns the `other` class.

## Application dataset

The application contains 15 annotated sequences reconstructed from match
video:

- six four-second clips;
- nine eight-second clips;
- synchronized 1280 × 720 broadcast video;
- synchronized 1050 × 590 canonical graph video;
- four tactical classes.

## Evaluation

Evaluation separates the following concerns:

- person and ball detection quality;
- track continuity and identity fragmentation;
- fixture-local team and `other` classification;
- canonical projection coverage and calibration quality;
- possession and attacking-direction reliability;
- tactical classification quality;
- retrieval relevance;
- human acceptance of the complete example.

Coverage and accuracy must be reported separately. Abstention, unresolved
tracks, and excluded clips should not be counted as correct predictions.

## Current limitations

- A player may be split across multiple track IDs.
- Small, occluded, or short-lived tracks may remain unresolved.
- Referees, staff, substitutes, and goalkeepers can be visually ambiguous.
- Broadcast calibration becomes less reliable near image and pitch boundaries.
- Ball observations can be sparse or incorrect.
- Possession and attacking direction depend on reconstructed geometry.
- Fixture-local validation data is limited.
- Tactical labels describe short dominant phases and may not capture
  counterpressing or rapid phase transitions.

## Reproducibility boundary

The public repository contains the product, reviewed catalogue metadata, and
this current methodology description. Raw footage, credentials, annotations,
evidence crops, model weights, generated graphs, review interfaces, and
fixture-specific diagnostics remain outside the public repository.

## Run the research pipeline

The research code is maintained on the private `dev` branch. Create an
isolated Python environment and install its dependencies:

```bash
git switch dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-research.txt
```

Run the complete research test suite:

```bash
python -m pytest -q
```

Research commands read manifests, footage, annotations, and model checkpoints
from the ignored local workspace. Inspect each command before supplying those
local paths:

```bash
python scripts/benchmark_player_tracking.py --help
python scripts/detect_track_ball.py --help
python scripts/train_graph_classifier.py --help
```

For example, run a detector/tracker experiment from local manifest and
experiment definitions:

```bash
python scripts/benchmark_player_tracking.py run \
  --manifest data/manifests/player-tracking-benchmark.json \
  --experiments data/manifests/player-tracking-experiments.json \
  --output data/logs/player-tracking
```

The `data/`, `models/`, `artifacts/`, and `third_party/` directories are
intentionally ignored. Do not commit footage, licensed provider data,
credentials, generated outputs, or model checkpoints.
