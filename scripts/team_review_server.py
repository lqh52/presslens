#!/usr/bin/env python3
"""Serve a local, NDA-safe review UI for player-track club labels."""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PressLens — team calibration</title>
<style>
:root{color-scheme:dark;font:14px Inter,ui-sans-serif,system-ui;background:#090e0b;color:#eff5f0}
*{box-sizing:border-box}body{margin:0}.app{max-width:1050px;margin:auto;padding:24px}
header{display:flex;justify-content:space-between;align-items:end;gap:18px;margin-bottom:18px}
h1{font-size:24px;margin:0}.muted{color:#94a198}.progress{text-align:right;font-variant-numeric:tabular-nums}
.panel{background:#121915;border:1px solid #29362f;border-radius:15px;padding:17px}
.crop{display:block;width:100%;background:#181c1a;border-radius:10px;min-height:260px;object-fit:contain}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}.chip{background:#1b2520;padding:8px 10px;border-radius:8px}
.suggestion{margin-top:14px;padding:11px;border:1px dashed #516258;border-radius:9px;color:#bcc8c0}
.saved{color:#b8f45e;font-weight:700}.complete{margin-top:14px;padding:11px;border-radius:9px;background:#1d3020;color:#c9f899;font-weight:700}
.actions{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:16px}
button{border:0;border-radius:10px;padding:14px 10px;font-weight:800;cursor:pointer;color:#fff}
.arsenal{background:#b52a2f}.burnley{background:#6d263d}.ignore{background:#48524d}.selected{outline:3px solid #b8f45e;outline-offset:2px}
.nav{display:flex;justify-content:space-between;align-items:center;margin-top:13px}
.nav button{background:#27332d;padding:9px 14px}.keys{color:#849189;font-size:12px;text-align:center}
@media(max-width:720px){header{align-items:start}.actions{grid-template-columns:1fr}.crop{min-height:180px}}
</style></head><body><main class="app">
<header><div><h1>Player team calibration</h1><div class="muted">Burnley vs Arsenal · local research labels</div></div><div class="progress" id="progress"></div></header>
<section class="panel">
 <img class="crop" id="crop" alt="Three crops from one player track">
 <div class="meta"><div class="chip" id="video"></div><div class="chip" id="track"></div><div class="chip" id="role"></div></div>
 <div class="suggestion" id="suggestion"></div>
 <div class="actions">
  <button class="arsenal" data-label="arsenal">A · Arsenal</button>
  <button class="burnley" data-label="burnley">B · Burnley</button>
  <button class="ignore" data-label="ignore">I · Exclude non-outfield</button>
 </div>
 <div class="complete" id="complete" hidden>All 125 labels are saved. Next now wraps to the first track for review.</div>
 <div class="nav"><button id="prev">← Previous</button><div class="keys">A / B / I label · ← / → navigate</div><button id="next">Next →</button></div>
</section></main>
<script>
let state,index=0;const $=id=>document.getElementById(id);
function current(){return state.items[index]}
function show(i){
 index=Math.max(0,Math.min(i,state.items.length-1));const x=current(),saved=state.labels[x.key]?.label;
 $("crop").src="/crops/"+encodeURIComponent(x.crop);$("video").textContent=x.video_id.toUpperCase();
 $("track").textContent=`Track ${x.track_id} · ${x.detections_in_window} detections`;
 $("role").textContent=x.role;$("suggestion").innerHTML=x.current_prediction
  ? `Current unsupervised suggestion: <b>${x.current_prediction}</b>. <span class="muted">Please judge the crops; this is not a ground-truth label.</span>`
  : `Current model abstained. <span class="muted">Use Ignore if the kit is not clear.</span>`;
 document.querySelectorAll("[data-label]").forEach(b=>b.classList.toggle("selected",b.dataset.label===saved));
 const done=state.reviewed===state.items.length;$("complete").hidden=!done;
 $("next").textContent=index===state.items.length-1?"First ↻":"Next →";
 $("progress").innerHTML=`<span class="${done?"saved":""}">${state.reviewed} / ${state.items.length} labeled</span><br><span class="muted">item ${index+1}</span>`;
}
function next(){show(index===state.items.length-1?0:index+1)}
async function label(value){
 const x=current(),wasNew=!state.labels[x.key];
 const r=await fetch("/api/label",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({key:x.key,label:value})});
 if(!r.ok){alert(`Save failed: ${await r.text()}`);return}
 state.labels[x.key]=await r.json();if(wasNew)state.reviewed++;show(Math.min(index+1,state.items.length-1));
}
document.querySelectorAll("[data-label]").forEach(b=>b.onclick=()=>label(b.dataset.label));
$("prev").onclick=()=>show(index-1);$("next").onclick=next;
document.addEventListener("keydown",e=>{
 if(e.repeat)return;const key=e.key.toLowerCase();
 if(key==="a")label("arsenal");else if(key==="b")label("burnley");else if(key==="i")label("ignore");
 else if(e.key==="ArrowLeft")show(index-1);else if(e.key==="ArrowRight")next();
});
fetch("/api/state").then(r=>r.json()).then(x=>{state=x;const first=x.items.findIndex(i=>!x.labels[i.key]);show(first<0?0:first)});
</script></body></html>"""


class TeamReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], directory: Path) -> None:
        self.directory = directory.resolve()
        self.crops = (directory / "crops").resolve()
        self.manifest_path = directory / "manifest.json"
        self.labels_path = directory / "labels.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.valid_keys = {item["key"] for item in self.manifest["items"]}
        self.lock = threading.Lock()
        super().__init__(address, TeamReviewHandler)

    def labels(self) -> dict:
        if not self.labels_path.exists():
            return {}
        return json.loads(self.labels_path.read_text()).get("labels", {})

    def save_label(self, key: str, label: str) -> dict:
        record = {
            "key": key,
            "label": label,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            labels = self.labels()
            labels[key] = record
            value = {"labels": labels}
            temporary = self.labels_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(value, indent=2) + "\n")
            temporary.replace(self.labels_path)
        return record


class TeamReviewHandler(BaseHTTPRequestHandler):
    server: TeamReviewServer

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            return
        if route == "/api/state":
            labels = self.server.labels()
            payload = {
                **self.server.manifest,
                "labels": labels,
                "reviewed": sum(
                    item["key"] in labels for item in self.server.manifest["items"]
                ),
            }
            self.send_bytes(json.dumps(payload).encode(), "application/json")
            return
        if route.startswith("/crops/"):
            requested = (self.server.crops / unquote(route[7:])).resolve()
            if requested.parent != self.server.crops or not requested.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_bytes(requested.read_bytes(), "image/jpeg")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/label":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("Invalid request size")
            value = json.loads(self.rfile.read(length))
            if value.get("key") not in self.server.valid_keys:
                raise ValueError("Unknown player track")
            if value.get("label") not in {"arsenal", "burnley", "ignore"}:
                raise ValueError("Invalid team label")
            saved = self.server.save_label(value["key"], value["label"])
            self.send_bytes(json.dumps(saved).encode(), "application/json")
        except (ValueError, json.JSONDecodeError) as error:
            self.send_bytes(
                str(error).encode(), "text/plain", HTTPStatus.BAD_REQUEST
            )

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("data/review/team_tracks"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    server = TeamReviewServer((args.host, args.port), args.directory)
    print(f"Team calibration UI: http://{args.host}:{args.port}")
    print("NDA imagery remains local; press Ctrl-C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
