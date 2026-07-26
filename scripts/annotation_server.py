#!/usr/bin/env python3
"""Local review UI for ranked SoccerNet clips.

The server binds to localhost by default, serves clips from an ignored directory,
and stores only human annotations in a JSON file.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PressLens — SoccerNet review</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui;background:#0b0f0d;color:#eef4ef}
*{box-sizing:border-box}body{margin:0}.app{max-width:1180px;margin:auto;padding:24px}
header{display:flex;align-items:end;justify-content:space-between;margin-bottom:18px}
h1{font-size:24px;margin:0}.muted{color:#94a39a;font-size:13px}.progress{font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(320px,1fr);gap:20px}
.panel{background:#121916;border:1px solid #27332d;border-radius:14px;padding:16px}
video{width:100%;aspect-ratio:16/9;background:#000;border-radius:9px}
.meta{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.chip{background:#1a241f;border-radius:8px;padding:10px;font-size:12px}
label{display:block;font-size:12px;color:#aab8b0;margin:0 0 5px}
select,textarea{width:100%;color:#eef4ef;background:#0d1310;border:1px solid #33443a;border-radius:8px;padding:9px;margin-bottom:12px}
textarea{min-height:70px;resize:vertical}.check{display:flex;gap:8px;align-items:center;margin-bottom:12px}
.check input{width:18px;height:18px}.check label{margin:0;color:#eef4ef}
.actions{display:flex;gap:8px}.actions button{flex:1;border:0;border-radius:9px;padding:11px;font-weight:700;cursor:pointer}
.secondary{background:#29352f;color:#eef4ef}.primary{background:#b5f55c;color:#111b0b}
#status{height:18px;margin-top:10px;color:#b5f55c;font-size:12px}
@media(max-width:800px){.grid{grid-template-columns:1fr}.meta{grid-template-columns:1fr}}
</style></head>
<body><main class="app">
<header><div><h1>PressLens review</h1><div class="muted">Local SoccerNet research annotation</div></div><div class="progress" id="progress"></div></header>
<div class="grid">
 <section class="panel"><video id="video" controls autoplay muted></video>
  <div class="meta"><div class="chip" id="match"></div><div class="chip" id="time"></div><div class="chip" id="score"></div></div>
  <p class="muted" id="query"></p>
 </section>
 <form class="panel" id="form">
  <div class="check"><input id="is_build_up" type="checkbox"><label for="is_build_up">Build-up under pressure is present</label></div>
  <label for="pressing_trigger">Pressing trigger</label><select id="pressing_trigger"></select>
  <label for="press_direction">Press direction</label><select id="press_direction"></select>
  <label for="central_option">Central option</label><select id="central_option"></select>
  <label for="outcome">Outcome</label><select id="outcome"></select>
  <label for="description">What happened?</label><textarea id="description" placeholder="Short tactical description"></textarea>
  <label for="notes">Research notes</label><textarea id="notes" placeholder="Ambiguity, camera limitations, useful cues…"></textarea>
  <div class="actions"><button type="button" class="secondary" id="prev">← Previous</button><button class="primary">Save & next →</button></div>
  <div id="status"></div>
 </form>
</div></main>
<script>
const fields={
 pressing_trigger:["none","back pass","goalkeeper reception","poor touch","wide reception","square pass","other"],
 press_direction:["none","toward left touchline","toward right touchline","inside","backward","mixed"],
 central_option:["not visible","open","screened","marked","used"],
 outcome:["forced wide","forced backward","forced long","turnover","played through","no clear outcome","not applicable"]
};
let state, index=0;
for(const [id,values] of Object.entries(fields)){const el=document.getElementById(id);for(const v of values)el.add(new Option(v,v))}
function stamp(s){const m=Math.floor(s/60),q=Math.floor(s%60);return `${m}:${String(q).padStart(2,"0")}`}
function show(i){
 index=Math.max(0,Math.min(i,state.items.length-1));const x=state.items[index],a=state.annotations[x.id]||{};
 video.src=x.clip_url;match.textContent=x.game;time.textContent=`Half ${x.half} · ${stamp(x.start_seconds)}`;
 score.textContent=`Rank ${index+1} · ${x.query_score.toFixed(3)}`;query.textContent=x.matched_query;
 progress.textContent=`${state.reviewed} / ${state.items.length} reviewed · clip ${index+1}`;
 is_build_up.checked=a.is_build_up||false;
 for(const id of Object.keys(fields))document.getElementById(id).value=a[id]||fields[id][0];
 description.value=a.description||"";notes.value=a.notes||"";status.textContent=a.updated_at?"Saved annotation":"";
}
async function save(e){e.preventDefault();const x=state.items[index],body={id:x.id,is_build_up:is_build_up.checked,description:description.value,notes:notes.value};
 for(const id of Object.keys(fields))body[id]=document.getElementById(id).value;
 const r=await fetch("/api/annotation",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});
 if(!r.ok){status.textContent=`Save failed: ${await r.text()}`;return}
 const saved=await r.json();if(!state.annotations[x.id])state.reviewed++;state.annotations[x.id]=saved;status.textContent="Saved";
 setTimeout(()=>show(index+1),180);
}
form.addEventListener("submit",save);prev.onclick=()=>show(index-1);
document.addEventListener("keydown",e=>{if(e.target.matches("textarea,select,input"))return;if(e.key==="ArrowRight")show(index+1);if(e.key==="ArrowLeft")show(index-1)});
fetch("/api/state").then(r=>r.json()).then(x=>{state=x;show(Math.max(0,x.items.findIndex(i=>!x.annotations[i.id])))});
</script></body></html>"""


class ReviewServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        manifest: Path,
        clips: Path,
        annotations: Path,
    ) -> None:
        self.manifest_path = manifest
        self.clips_path = clips.resolve()
        self.annotations_path = annotations
        super().__init__(address, ReviewHandler)

    def annotations(self) -> dict:
        if not self.annotations_path.exists():
            return {}
        return json.loads(self.annotations_path.read_text()).get("annotations", {})

    def save_annotation(self, annotation: dict) -> dict:
        from datetime import datetime, timezone

        annotations = self.annotations()
        annotation["updated_at"] = datetime.now(timezone.utc).isoformat()
        annotations[annotation["id"]] = annotation
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        self.annotations_path.write_text(
            json.dumps({"annotations": annotations}, indent=2) + "\n"
        )
        return annotation


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            return
        if route == "/api/state":
            payload = json.loads(self.server.manifest_path.read_text())
            annotations = self.server.annotations()
            clips = sorted(self.server.clips_path.glob("*.mp4"))
            clip_by_id = {
                clip.stem.split("_", 1)[1]: f"/clips/{clip.name}"
                for clip in clips
                if "_" in clip.stem
            }
            items = [
                {**item, "clip_url": clip_by_id[item["id"]]}
                for item in payload["candidates"]
                if item["id"] in clip_by_id
            ]
            body = {
                "items": items,
                "annotations": annotations,
                "reviewed": sum(item["id"] in annotations for item in items),
            }
            self.send_bytes(json.dumps(body).encode(), "application/json")
            return
        if route.startswith("/clips/"):
            requested = (self.server.clips_path / unquote(route[7:])).resolve()
            if requested.parent != self.server.clips_path or not requested.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_bytes(
                requested.read_bytes(),
                mimetypes.guess_type(requested.name)[0] or "video/mp4",
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/annotation":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            annotation = json.loads(self.rfile.read(length))
            valid_ids = {
                item["id"]
                for item in json.loads(self.server.manifest_path.read_text())[
                    "candidates"
                ]
            }
            if annotation.get("id") not in valid_ids:
                raise ValueError("Unknown candidate id")
            saved = self.server.save_annotation(annotation)
            self.send_bytes(json.dumps(saved).encode(), "application/json")
        except (ValueError, json.JSONDecodeError) as error:
            self.send_bytes(str(error).encode(), "text/plain", HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/ranked_candidates.json"),
    )
    parser.add_argument(
        "--clips", type=Path, default=Path("data/clips/candidates")
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/reviewed.json"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ReviewServer(
        (args.host, args.port), args.manifest, args.clips, args.annotations
    )
    print(f"Review UI: http://{args.host}:{args.port}")
    print("NDA video remains local; press Ctrl-C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
