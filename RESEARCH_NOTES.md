# PressLens research notes

## Reading the evidence

Each result pairs an annotated broadcast clip with a canonical pitch graph.
The two videos are synchronized and represent the same frames.

The canonical graph shows:

- one colour for each team;
- the reconstructed ball position;
- same-team edges summarizing local team structure;
- cross-team edges for short pressure relationships;
- track IDs for inspection.

Only resolved team tracks participate in tactical graph structure. Unresolved
and non-team detections are not used as team nodes.

## Tactical class definitions

### High press — wing

The defending team engages high up the pitch and concentrates pressure near a
touchline. The ball, nearby defenders, and the possession team’s available
options should support a wide pressing interpretation.

### High press — central

The defending team engages high through central build-up lanes. Pressure is
focused around central progression rather than primarily steering play toward
a flank.

### Medium press

The defending team engages around the middle third. The shape is higher than a
deep block but does not consistently press the possession team’s first
build-up line.

### Low block

The defending team is compact near its own penalty area, with most of its
outfield shape behind the ball. The label describes collective depth and
compactness, not merely one defender standing deep.

## Interpretation

- Classification confidence is the graph model’s probability for the selected
  sequence.
- Retrieval score measures text relevance, not tactical certainty.
- The representative frame is evidence for the sequence, not a standalone
  event label.
- Graph edges are explanatory structure, not observed passes.
- Missing nodes can reflect tracking, identity, or projection uncertainty.
- A plausible canonical graph can still contain an incorrect individual track.

## Application dataset

The current product contains 15 annotated examples across four classes. Six
examples are four seconds long and nine are eight seconds long. Broadcast
videos are served at 1280 × 720 and canonical videos at 1050 × 590, all at
25 frames per second.
