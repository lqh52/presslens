# PressLens research notes

PressLens retrieves short tactical situations reconstructed from football
broadcast video. A text-embedding model encodes the user query and each
situation description. Retrieval combines normalized MiniLM cosine similarity
(65%) with BM25 lexical relevance (35%) to rank the results.

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

1. Open `http://127.0.0.1:4173`.
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
direction are reconstructed model outputs. Referees, substitutes, goalkeepers,
off-pitch detections, and unstable tracks are filtered conservatively, but
errors can remain. Broadcast cuts and uncertain pitch calibration can also
reduce temporal consistency.
