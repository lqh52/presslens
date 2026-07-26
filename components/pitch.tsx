"use client";

import type { Clip } from "@/lib/types";

export function Pitch({ clip, compact = false }: { clip: Clip; compact?: boolean }) {
  return (
    <svg
      className={`pitch ${compact ? "pitch--compact" : ""}`}
      viewBox="0 0 100 68"
      role="img"
      aria-label={`Tactical reconstruction for ${clip.id}`}
    >
      <defs>
        <marker id={`arrow-build-${clip.id}`} markerWidth="5" markerHeight="5" refX="3" refY="2.5" orient="auto">
          <path d="M0,0 L0,5 L5,2.5 z" fill="#e7efe9" />
        </marker>
        <marker id={`arrow-press-${clip.id}`} markerWidth="5" markerHeight="5" refX="3" refY="2.5" orient="auto">
          <path d="M0,0 L0,5 L5,2.5 z" fill="#ff5b45" />
        </marker>
      </defs>
      <rect x="1" y="1" width="98" height="66" rx="1" className="pitch-line" />
      <line x1="50" y1="1" x2="50" y2="67" className="pitch-line" />
      <circle cx="50" cy="34" r="9.15" className="pitch-line" />
      <circle cx="50" cy="34" r=".8" className="pitch-fill" />
      <rect x="1" y="13.8" width="16.5" height="40.4" className="pitch-line" />
      <rect x="82.5" y="13.8" width="16.5" height="40.4" className="pitch-line" />
      <rect x="1" y="24.8" width="5.5" height="18.4" className="pitch-line" />
      <rect x="93.5" y="24.8" width="5.5" height="18.4" className="pitch-line" />
      {clip.players.map((player, index) => (
        <g key={`${clip.id}-${index}`}>
          <line
            x1={player.x}
            y1={player.y}
            x2={player.x + player.dx}
            y2={player.y + player.dy}
            className={`motion motion--${player.team}`}
            markerEnd={`url(#arrow-${player.team}-${clip.id})`}
          />
          <circle cx={player.x} cy={player.y} r="2.25" className={`player player--${player.team}`} />
          <text x={player.x} y={player.y + 0.9} className="player-number">
            {player.role === "goalkeeper" ? "G" : index + 1}
          </text>
          {player.controlsBall && <circle cx={player.x} cy={player.y} r="3.15" className="controller-ring" />}
        </g>
      ))}
      {clip.ball && <circle cx={clip.ball.x} cy={clip.ball.y} r="1.15" className="ball" />}
    </svg>
  );
}
