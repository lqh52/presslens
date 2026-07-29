export type Situation =
  | "high_press_wing"
  | "high_press_central"
  | "medium_press"
  | "low_block";

export type Player = {
  x: number;
  y: number;
  dx: number;
  dy: number;
  team: "press" | "build";
  role: "player" | "goalkeeper";
  controlsBall: boolean;
};

export type Clip = {
  id: string;
  match: string;
  minute: string;
  half: 1 | 2;
  videoId: string;
  video: string;
  canonicalImage: string;
  canonicalVideo: string;
  timeSeconds: number;
  durationSeconds: number;
  frame: number;
  situation: Situation;
  title: string;
  confidence: number;
  majorityFrames: number;
  validFrames: number;
  orientationValidated: boolean;
  reviewDecision: "include" | "exclude" | "unreviewed";
  labelSource: "expert_override" | "expert_review" | "graph_classifier";
  phase: string;
  visibleNodes: number;
  possessionConfident: boolean;
  ballConfidence: number;
  possessionClub: string;
  pressingClub: string;
  attackDirection: "left_to_right" | "right_to_left" | "undetermined";
  directionSource: string;
  directionConfidence?: number;
  teamIdentityMap: Record<string, string>;
  ballHolderDistanceM: number | null;
  description: string;
  evidence: string[];
  tags: string[];
  probabilities: Record<Situation, number>;
  weakLabel: string;
  weakRule: string;
  thumbnail: string;
  players: Player[];
  ball: { x: number; y: number } | null;
  overlayTrackFilter?: {
    mode: string;
    excluded_track_ids: number[];
  } | null;
};

export type DemoManifest = {
  name: string;
  source: string;
  count: number;
  videoCount: number;
  matchCount?: number;
  reviewStatus?: string;
  videos: Array<{ id: string; half: 1 | 2; startSeconds: number; path: string }>;
  clips: Clip[];
};
