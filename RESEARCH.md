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

## Reproducing the research

This section is for researchers cloning the repository. It separates examples
that run from a fresh clone from experiments that require licensed football
video or external game-state reconstruction software.

### System requirements

- Git
- Python 3.11
- `ffmpeg` and `ffprobe` for video experiments
- A CUDA GPU is optional for the quickstarts and recommended for video ranking
  and model training

Create the Python environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-research.txt
python -m pytest -q
```

The tested suite contains unit tests that construct temporary inputs; it does
not require football footage or pretrained project checkpoints.

### Quickstart A: train the synthetic graph baseline

This experiment requires no external data and downloads no model weights. It
generates controlled player-and-ball graphs and trains the tactical graph
classifier:

```bash
python scripts/generate_synthetic_graphs.py \
  --output data/graphs/synthetic.npz \
  --samples 5000

python scripts/train_graph_classifier.py \
  --data data/graphs/synthetic.npz \
  --output models/tactical_graph_net.pt \
  --epochs 5
```

The first command creates the training set. The second creates
`models/tactical_graph_net.pt` and a metrics JSON file. Both locations are
excluded from Git because they are generated outputs, not missing repository
inputs.

### Quickstart B: track people in any MP4

Edit `examples/research/clips.example.json` so `video` points to an MP4 on your
machine. Then run:

```bash
python scripts/benchmark_player_tracking.py run \
  --manifest examples/research/clips.example.json \
  --experiments examples/research/player-tracking-experiments.json \
  --output data/benchmarks/player-tracking \
  --device cpu

python scripts/benchmark_player_tracking.py summarize \
  --results data/benchmarks/player-tracking \
  --output data/benchmarks/player-tracking-summary.json
```

The example uses the official Ultralytics `yolo26n.pt` checkpoint. Ultralytics
downloads it automatically on first use. This baseline tracks COCO `person`
detections; it does not yet distinguish players, goalkeepers, referees, or
staff, and it does not detect the ball. Those are research targets rather than
claims of the baseline.

Ultralytics publishes YOLO26 weights and documents their AGPL-3.0 and
enterprise licensing options:
<https://docs.ultralytics.com/models/yolo26/>.

### Football datasets

#### SoccerNet broadcast video

SoccerNet video is not redistributed by this repository. Researchers must
complete the SoccerNet NDA and receive a video password:
<https://www.soccer-net.org/data>. SoccerNet states that the dataset is for
research and not commercial use:
<https://www.soccer-net.org/faq>.

After access is approved, download one match at both resolutions. Replace the
match identifier with a path listed by SoccerNet:

```bash
python scripts/download_soccernet_matches.py \
  --match "train=CHAMPIONSHIP/SEASON/GAME" \
  --resolution 224p

python scripts/download_soccernet_matches.py \
  --match "train=CHAMPIONSHIP/SEASON/GAME" \
  --resolution 720p
```

The downloader prompts for the password without writing it to disk and stores
the videos and `Labels-v2.json` below `data/raw/soccernet/`.

Create uniformly sampled in-play candidate windows:

```bash
python scripts/build_candidates.py \
  --game-dir "data/raw/soccernet/CHAMPIONSHIP/SEASON/GAME" \
  --output data/manifests/candidates.json
```

Rank candidates with X-CLIP:

```bash
python scripts/rank_video_candidates.py \
  --manifest data/manifests/candidates.json \
  --output data/manifests/ranked_candidates.json \
  --limit 100 \
  --top-k 20
```

`microsoft/xclip-base-patch32` is downloaded automatically from Hugging Face
on first use. Ranking is practical on a CUDA GPU and can be slow on CPU.

#### Other supported sources

- `scripts/build_statsbomb_pressure_maps.py` downloads events and 360 data for
  an explicitly supplied StatsBomb match ID. Follow the attribution and
  licensing requirements in <https://github.com/statsbomb/open-data>.
- `scripts/build_skillcorner_pressing_samples.py` consumes match metadata,
  dynamic events, phase labels, and tracking files obtained from
  <https://github.com/SkillCorner/opendata>. Supply those files through the
  script's required command-line arguments.

### Model acquisition

| Model | Acquisition |
| --- | --- |
| YOLO26 | Use an official filename such as `yolo26n.pt`; Ultralytics downloads it on first use. |
| X-CLIP | `rank_video_candidates.py` downloads `microsoft/xclip-base-patch32` from Hugging Face. |
| MiniLM text encoder | Run `python scripts/download_text_embedding_model.py`. |
| DINOv2-small | `extract_dino_track_features.py` downloads `facebook/dinov2-small` from Hugging Face. |
| Tactical graph model | Generate data and train it with Quickstart A; no checkpoint is supplied. |
| Team/role models | Train them from reviewed fixture data with `train_team_identity.py` and `classify_track_identities.py`; they are fixture-specific. |

The local Qwen and Gemini labeling scripts are optional annotation workflows,
not prerequisites for either quickstart. Gemini requires one of
`AGENT_PLATFORM_API`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY`. The local Qwen
workflow downloads a large checkpoint and requires suitable GPU memory.

### Broadcast video to canonical pitch coordinates

This advanced path runs the official
[SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate)
(GSR) baseline on a video, then converts its TrackLab state into PressLens
graphs. The result contains tracked people, roles, team-side clusters, camera
calibration, and player and ball locations in canonical 105 x 68 metre pitch
coordinates.

GSR has its own Python 3.9 environment and GPU dependencies. Keep it separate
from the PressLens Python 3.11 environment. From the PressLens repository root:

```bash
mkdir -p third_party
git clone https://github.com/SoccerNet/sn-gamestate.git \
  third_party/sn-gamestate

cd third_party/sn-gamestate
uv venv --python 3.9
uv pip install -e .
uv run mim install mmcv==2.0.1
cd ../..
```

Install `uv` first if it is unavailable:
<https://docs.astral.sh/uv/getting-started/installation/>. These commands
follow the upstream GSR installation guide. GSR requires a CUDA-capable GPU;
reduce module batch sizes in
`third_party/sn-gamestate/sn_gamestate/configs/soccernet.yaml` if GPU memory is
insufficient.

Verify the external installation before using PressLens:

```bash
third_party/sn-gamestate/.venv/bin/tracklab --help
```

The upstream baseline downloads its pretrained detector, PRTReID, calibration,
and other model files on first use. The expected files subsequently include:

```text
third_party/sn-gamestate/pretrained_models/
  calibration/
  reid/prtreid-soccernet-baseline.pth.tar
  yolo/yolo11m.pt
```

The upstream validation example can be used to trigger and verify all
automatic downloads:

```bash
cd third_party/sn-gamestate
uv run tracklab -cn soccernet
cd ../..
```

That upstream command may also download the SoccerNet-GSR benchmark dataset.
For an arbitrary local football video, edit
`examples/research/gsr-clips.example.json` and set `clip_path` to the absolute
MP4 path. `nframes` may be omitted because PressLens uses `ffprobe` to obtain
it.

Validate the paths and show the TrackLab command without starting inference:

```bash
python scripts/run_gsr_batch.py \
  --manifest examples/research/gsr-clips.example.json \
  --gpu-ids 0 \
  --dry-run
```

Run GSR:

```bash
python scripts/run_gsr_batch.py \
  --manifest examples/research/gsr-clips.example.json \
  --gpu-ids 0
```

The batch runner writes logs to `data/logs/gsr-batch/` and the TrackLab state
below:

```text
third_party/sn-gamestate/outputs/local-demo/<date>/<time>/states/local-demo.pklz
```

It validates completed states and resumes without recomputing valid results.
For multiple GPUs, use a comma-separated list such as `--gpu-ids 0,1`.

Locate the exact state produced by the preceding run:

```bash
find third_party/sn-gamestate/outputs/local-demo \
  -path '*/states/local-demo.pklz' -print
```

Convert it into PressLens graphs. Replace `<state-path>` with the path printed
above:

```bash
third_party/sn-gamestate/.venv/bin/python \
  scripts/convert_tracklab_state.py \
  --state "<state-path>" \
  --video "/absolute/path/to/football-clip.mp4" \
  --yolo third_party/sn-gamestate/pretrained_models/yolo/yolo11m.pt \
  --output data/graphs/local-demo.npz \
  --sequence-id local-demo \
  --neutral-team-names \
  --disable-team-labels \
  --disable-team-model
```

This command uses the GSR camera calibration and person tracks, runs the
upstream YOLO checkpoint for COCO `sports ball`, and creates:

- `data/graphs/local-demo.npz`: graph tensors and visibility masks;
- `data/graphs/local-demo.jsonl`: per-frame pitch coordinates, possession,
  direction provenance, ball confidence, and team-side metadata.

Neutral names deliberately report `Team A` and `Team B`; they do not claim
club identity. Researchers can replace them with reviewed identities using
the converter's `--match-registry` and `--match-id` options.

Derive auditable heuristic tactical labels from the reconstructed graph:

```bash
python scripts/derive_weak_tactical_labels.py \
  --graphs data/graphs/local-demo.npz \
  --output data/graphs/local-demo-weak-labels.npz
```

`process_gsr_outputs.py` automates conversion, direction calibration, weak
labeling, and classification for a multi-match experiment, but it additionally
requires a reviewed team registry, at least two clips per match-half for
direction voting, and a trained tactical checkpoint. Use the direct conversion
above for a first external reproduction. The official upstream README is the
source of truth for GSR installation, model downloads, supported CUDA
versions, and dataset version changes.

### Files created locally

The following directories are created by the commands above:

- `data/`: downloaded, licensed, derived, or generated datasets;
- `models/`: downloaded or trained checkpoints;
- `artifacts/`: reports and rendered research outputs;
- `third_party/`: separately installed external research systems.

They are excluded from Git to prevent redistribution of licensed footage,
large generated files, credentials, and machine-specific external
dependencies. The committed examples under `examples/research/` define the
input schemas needed by the public quickstart.
