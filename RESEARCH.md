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

## Supervision

Fixture-local examples provide identity context and validation. Conservative
recovery is applied only to unresolved tracks. Tactical supervision is learned
from canonical spatiotemporal graphs rather than broadcast appearance.

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
- The published catalogue is small and not class-balanced enough for broad
  performance claims.
- Tactical labels describe short dominant phases and may not capture
  counterpressing or rapid phase transitions.

## Reproducibility boundary

The public repository contains the product, reviewed catalogue metadata, and
this current methodology description. Raw footage, credentials, annotations,
evidence crops, model weights, generated graphs, review interfaces, and
fixture-specific diagnostics remain outside the public repository.
