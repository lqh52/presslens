#!/usr/bin/env python3
"""Review 20 repaired video projections and save pressing-phase labels locally."""

from __future__ import annotations

import argparse
import json
import mimetypes
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


LABELS = [
    "high_press_wing",
    "high_press_central",
    "medium_press",
    "low_block",
    "exclude",
]
HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PressLens — pressing review</title><style>
:root{color-scheme:dark;font:14px Inter,system-ui;background:#09110c;color:#edf6ef}*{box-sizing:border-box}
body{margin:0}.app{max-width:1500px;margin:auto;padding:18px}header{display:flex;justify-content:space-between;gap:16px;align-items:end;margin-bottom:14px}
h1{font-size:22px;margin:0 0 4px}.muted{color:#91a199;font-size:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.panel{background:#111b15;border:1px solid #2b3c31;border-radius:12px;padding:12px;min-width:0}video,canvas{display:block;width:100%;aspect-ratio:16/9;background:#000;border-radius:8px}
canvas{background:#286b34}.bar{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.chip{background:#1d2a22;border:1px solid #34483a;border-radius:99px;padding:5px 8px;font-size:11px}
.prediction{font-size:20px;font-weight:800;margin:4px 0}.probs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:9px 0}.prob{background:#0b130e;border-radius:8px;padding:8px}
.prob span{display:block;color:#91a199;font-size:10px}.help{line-height:1.45;color:#b3c0b8;border-left:2px solid #d6aa55;padding-left:10px;margin:10px 0}
.labels{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.labels button,.nav button{border:1px solid #405748;background:#223128;color:#eef6ef;border-radius:9px;padding:11px;font-weight:750;cursor:pointer}
.labels button.selected{background:#b9f46a;color:#10200b;border-color:#b9f46a}.labels button.exclude{background:#4b292c;border-color:#704047}
textarea{width:100%;height:62px;margin-top:9px;background:#0a120d;color:#eef6ef;border:1px solid #3b5042;border-radius:8px;padding:8px}
.nav{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.status{min-height:18px;color:#b9f46a;margin-top:7px;font-size:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.app{padding:10px}} </style></head>
<body><main class="app"><header><div><h1>20-video pressing review</h1><div class="muted">Synchronized broadcast and repaired canonical projection · labels save locally</div></div><div id="progress"></div></header>
<div class="grid"><section class="panel"><video id="video" controls playsinline preload="metadata"></video><div class="bar"><span class="chip" id="clip"></span><span class="chip" id="frame"></span><span class="chip" id="calibration"></span></div></section>
<section class="panel"><canvas id="pitch" width="1050" height="590"></canvas></section>
<section class="panel"><div class="muted">Graph-only proposal</div><div class="prediction" id="prediction"></div><div class="probs" id="probs"></div>
<div class="help"><b>High press wing:</b> pressure high up the pitch with the ball/press concentrated near a flank.<br><b>High press central:</b> pressure high up the pitch through the central channel.<br><b>Medium press:</b> pressing shape engages around the middle third rather than the opponent's first line.<br><b>Low block:</b> the defending team is compact near its own penalty area, with most of its outfield shape behind the ball.<br><b>Exclude:</b> wrong projection, identity, ball, possession, counterpress, or no clear phase.</div></section>
<section class="panel"><div class="labels" id="labels"></div><textarea id="notes" placeholder="Optional note: projection, ball, identity, or tactical ambiguity"></textarea>
<div class="nav"><button id="back">← Previous</button><button id="next">Next →</button></div><div class="status" id="status"></div></section></div></main>
<script>
const $=x=>document.getElementById(x), titles={high_press_wing:"High press — wing",high_press_central:"High press — central",medium_press:"Medium press",low_block:"Low block",exclude:"Exclude / unsure"};
let state,index=0,detail=null;
const xy=(x,y)=>[(x+52.5)*10,(y+34)*8.676], pct=x=>`${Math.round(100*x)}%`;
function paintPitch(){const c=$("pitch"),x=c.getContext("2d");x.clearRect(0,0,c.width,c.height);x.fillStyle="#286b34";x.fillRect(0,0,c.width,c.height);x.strokeStyle="#e6eee7";x.lineWidth=3;x.strokeRect(4,4,1042,582);x.beginPath();x.moveTo(525,4);x.lineTo(525,586);x.stroke();x.beginPath();x.arc(525,295,79,0,7);x.stroke();x.strokeRect(4,153,165,284);x.strokeRect(881,153,165,284)}
function draw(){if(!detail)return;paintPitch();const v=$("video"),n=detail.frames.length||1,frame=Math.min(n-1,Math.max(0,Math.round((v.currentTime/(v.duration||1))*(n-1)))),row=detail.frames[frame],x=$("pitch").getContext("2d");$("frame").textContent=`Frame ${frame+1} / ${n}`;$("calibration").textContent=(row.calibration?.projection_method||"unreliable").replaceAll("_"," ");
for(const o of row.objects){const team=detail.teams[String(o.track_id)]||"unknown";if(!["team_a","team_b"].includes(team)||Math.abs(o.x)>52.5||Math.abs(o.y)>34)continue;const p=xy(o.x,o.y);x.fillStyle=team==="team_a"?"#ff6d61":"#55a7ff";x.beginPath();x.arc(...p,9,0,7);x.fill();x.fillStyle="#fff";x.font="11px sans-serif";x.fillText(o.track_id,p[0]+10,p[1]-8)}
if(row.ball){const p=xy(...row.ball);x.fillStyle="#ffe444";x.beginPath();x.arc(...p,7,0,7);x.fill();x.strokeStyle="#111";x.lineWidth=2;x.stroke()}}
async function show(i){index=(i+state.items.length)%state.items.length;const item=state.items[index];detail=await fetch(`/api/clip/${encodeURIComponent(item.clip_id)}`).then(r=>r.json());$("clip").textContent=item.clip_id;$("prediction").textContent=`${titles[item.label]} · ${pct(item.confidence)}`;$("probs").innerHTML=Object.entries(item.probabilities).map(([k,v])=>`<div class="prob"><span>${titles[k]}</span><b>${pct(v)}</b></div>`).join("");$("video").src=`/video/${encodeURIComponent(item.clip_id)}`;$("video").load();$("notes").value=state.annotations[item.clip_id]?.notes||"";select(state.annotations[item.clip_id]?.label||"");progress();draw()}
function select(label){for(const b of $("labels").children)b.classList.toggle("selected",b.dataset.label===label)}
async function save(label){const item=state.items[index],body={clip_id:item.clip_id,label,notes:$("notes").value};const r=await fetch("/api/annotation",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});if(!r.ok){$("status").textContent=await r.text();return}state.annotations[item.clip_id]=await r.json();select(label);$("status").textContent="Saved";progress();setTimeout(()=>show(index+1),180)}
function progress(){const done=Object.keys(state.annotations).length;$("progress").textContent=`${done} / ${state.items.length} reviewed · ${index+1} of ${state.items.length}`}
for(const label of ["high_press_wing","high_press_central","medium_press","low_block","exclude"]){const b=document.createElement("button");b.textContent=titles[label];b.dataset.label=label;b.className=label==="exclude"?"exclude":"";b.onclick=()=>save(label);$("labels").append(b)}
$("back").onclick=()=>show(index-1);$("next").onclick=()=>show(index+1);$("video").ontimeupdate=draw;$("video").onloadedmetadata=draw;
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;if(e.key==="1")save("high_press_wing");if(e.key==="2")save("high_press_central");if(e.key==="3")save("medium_press");if(e.key==="4")save("low_block");if(e.key==="5")save("exclude");if(e.key==="ArrowLeft")show(index-1);if(e.key==="ArrowRight")show(index+1)});
fetch("/api/state").then(r=>r.json()).then(s=>{state=s;const u=s.items.findIndex(i=>!s.annotations[i.clip_id]);show(u<0?0:u)});
</script></body></html>"""


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], directory: Path) -> None:
        self.directory = directory.resolve()
        predictions = json.loads(
            (self.directory / "pressing-review-predictions.json").read_text()
        )["clips"]
        self.items = [
            {
                "clip_id": row["clip_id"],
                "label": row["summary"]["label"],
                "confidence": row["summary"]["confidence"],
                "probabilities": row["summary"]["probabilities"],
                "windows": row["summary"]["windows"],
            }
            for row in predictions
        ]
        self.ids = {row["clip_id"] for row in self.items}
        self.annotation_path = self.directory / "pressing-review-labels.json"
        self.lock = threading.Lock()
        super().__init__(address, Handler)

    def annotations(self) -> dict:
        if not self.annotation_path.exists():
            return {}
        return json.loads(self.annotation_path.read_text()).get("labels", {})

    def video(self, clip_id: str) -> Path:
        data = json.loads((self.directory / "ball-tracking" / f"{clip_id}.json").read_text())
        return Path(data["video_path"]).resolve()

    def detail(self, clip_id: str) -> dict:
        projection = json.loads(
            (self.directory / "pitch-projections" / f"{clip_id}.json").read_text()
        )
        balls = json.loads(
            (self.directory / "ball-tracking" / f"{clip_id}.json").read_text()
        )["frames"]
        identity = json.loads(
            (self.directory / "identities" / f"{clip_id}.json").read_text()
        )
        teams = {str(row["track_id"]): row.get("label", "other") for row in identity["tracks"]}
        frames = []
        for row, ball_row in zip(projection["frames"], balls):
            ball = ball_row.get("ball")
            frames.append(
                {
                    "frame": row["frame"],
                    "calibration": row.get("calibration", {}),
                    "objects": row.get("objects", []),
                    "ball": ball.get("pitch_xy") if ball else None,
                }
            )
        return {"clip_id": clip_id, "teams": teams, "frames": frames}


class Handler(BaseHTTPRequestHandler):
    server: Server

    def send(self, body: bytes, kind: str, status: int = 200, **headers: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send(HTML.encode(), "text/html; charset=utf-8")
        elif route == "/api/state":
            self.send(
                json.dumps(
                    {"items": self.server.items, "annotations": self.server.annotations()}
                ).encode(),
                "application/json",
            )
        elif route.startswith("/api/clip/"):
            clip_id = unquote(route[10:])
            if clip_id not in self.server.ids:
                self.send_error(404)
                return
            self.send(json.dumps(self.server.detail(clip_id)).encode(), "application/json")
        elif route.startswith("/video/"):
            clip_id = unquote(route[7:])
            if clip_id not in self.server.ids:
                self.send_error(404)
                return
            self.send_video(self.server.video(clip_id))
        else:
            self.send_error(404)

    def send_video(self, path: Path) -> None:
        size = path.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        value = self.headers.get("Range")
        if value and value.startswith("bytes="):
            left, _, right = value[6:].partition("-")
            start = int(left) if left else 0
            end = min(int(right), end) if right else end
            status = HTTPStatus.PARTIAL_CONTENT
        with path.open("rb") as handle:
            handle.seek(start)
            body = handle.read(end - start + 1)
        headers = {"Accept_Ranges": "bytes"}
        if status == HTTPStatus.PARTIAL_CONTENT:
            headers["Content_Range"] = f"bytes {start}-{end}/{size}"
        self.send(
            body,
            mimetypes.guess_type(path.name)[0] or "video/mp4",
            status,
            **headers,
        )

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/annotation":
            self.send_error(404)
            return
        try:
            row = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            clip_id, label = str(row["clip_id"]), str(row["label"])
            if clip_id not in self.server.ids or label not in LABELS:
                raise ValueError("Invalid clip or label")
            notes = str(row.get("notes", ""))[:4000]
            saved = {
                "clip_id": clip_id,
                "label": label,
                "notes": notes,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with self.server.lock:
                labels = self.server.annotations()
                labels[clip_id] = saved
                atomic_json(
                    self.server.annotation_path,
                    {"schema_version": 1, "labels": labels},
                )
            self.send(json.dumps(saved).encode(), "application/json")
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.send(str(error).encode(), "text/plain", 400)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("artifacts/tactical-coverage-review")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    server = Server((args.host, args.port), args.directory)
    print(f"Pressing review: http://{args.host}:{args.port}", flush=True)
    print(f"Labels: {server.annotation_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
