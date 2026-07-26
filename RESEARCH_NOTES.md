# PressLens research notes

PressLens retrieves short tactical situations reconstructed from football
broadcast video. A text-embedding model encodes the user query and each
situation description. The deployed system uses quantized
`Snowflake/snowflake-arctic-embed-s` embeddings with CLS pooling and its
retrieval query prefix. It combines normalized semantic similarity (90%) with
BM25 lexical relevance (10%) to rank the results.

Focused intent guards treat explicit negative-pressure language as no local
pressure and abstain for unsupported concepts such as corners, red cards, and
low blocks. These guards are deliberately narrow; they are not a replacement
for the tactical classifier.

## Supported tactical classes

### Central screen

The defending shape occupies central progression lanes ahead of the ball,
discouraging or blocking a direct pass through the middle.

### High press

Several defenders compress the ball area while the possession team builds in
its defensive third.

### No local pressure

The reconstructed state does not show coordinated pressure close to the ball.
The nearest pressure is distant or the local pressing structure is unclear.

Touchline-trap classes are not currently exposed as supported classes because
the reviewed dataset does not contain retained examples of them.

## Using the app

1. Open the [published application](https://lqh52.github.io/presslens/) or the
   local application at `http://127.0.0.1:3000`.
2. Describe a tactical situation in the search field, or select a suggested
   query.
3. Optionally filter results by tactical class or reliable possession.
4. Select a result to open the synchronized broadcast and canonical pitch
   videos.
5. Interpret the hybrid retrieval score as text relevance, not probability.
6. Interpret classification confidence as the graph classifier's class
   probability for the selected excerpt.
7. Inspect frame agreement, geometric evidence, attacking direction, and the
   full class-probability distribution before drawing a conclusion.

## Current limitations

Player locations, team identity, ball position, possession, and attacking
direction are reconstructed model outputs. The published broadcast overlay
retains tracked detections at or above 45% confidence; non-team detections are
shown separately and do not participate in team-structure edges. Manual track
labels override inferred identities where available. Errors can remain, and
broadcast cuts or uncertain pitch calibration can reduce temporal consistency.

The public catalogue contains eight reviewed four-second videos and three
supported classes. Clips within the same class currently share short catalogue
descriptions, so retrieval is more meaningful at class level than for ordering
near-identical examples within a class. Synthetic tactical training,
touchline-trap experiments, and labeling tools are not published.
