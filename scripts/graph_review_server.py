#!/usr/bin/env python3
"""Local blinded expert-review UI for SoccerNet tactical graphs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PressLens expert review</title><style>
:root{color-scheme:dark;font:14px system-ui;background:#0b100d;color:#eef5ef}*{box-sizing:border-box}body{margin:0}
main{max-width:1250px;margin:auto;padding:22px}header{display:flex;justify-content:space-between;align-items:end;margin-bottom:16px}
h1{margin:0;font-size:23px}.muted{color:#91a097}.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}
.panel{background:#131a16;border:1px solid #2b3930;border-radius:13px;padding:14px}img{width:100%;border-radius:8px}
canvas{width:100%;aspect-ratio:105/68;background:#204d2d;border-radius:8px;margin-top:10px}
label{display:block;color:#a9b5ae;margin:11px 0 5px}select,textarea{width:100%;background:#0c120e;color:#fff;border:1px solid #3a4b40;border-radius:8px;padding:10px}
textarea{height:80px}.checks{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.checks label{margin:0;color:#eef5ef}
.actions{display:flex;gap:8px;margin-top:14px}button{flex:1;border:0;border-radius:8px;padding:11px;font-weight:700}.save{background:#b6f45f;color:#102000}.prev{background:#2c3931;color:white}
@media(max-width:850px){.grid{grid-template-columns:1fr}}</style></head>
<body><main><header><div><h1>Blinded tactical review</h1><div class="muted">Broadcast frame + canonical graph</div></div><div id="progress"></div></header>
<div class="grid"><section class="panel"><img id="frame"><canvas id="pitch" width="840" height="544"></canvas><div class="muted" id="meta"></div></section>
<form class="panel" id="form"><label>Situation</label><select id="situation"></select>
<div class="checks"><label><input type="checkbox" id="build_up"> Build-up phase</label><label><input type="checkbox" id="press_present"> Organized press</label>
<label><input type="checkbox" id="orientation_valid" checked> Orientation valid</label><label><input type="checkbox" id="graph_sufficient" checked> Graph sufficient</label></div>
<label>Confidence</label><select id="confidence"><option>high</option><option selected>medium</option><option>low</option></select>
<label>Rationale</label><textarea id="rationale" placeholder="State the visible geometric evidence"></textarea>
<div class="actions"><button type="button" class="prev" id="prev">← Previous</button><button class="save">Save & next →</button></div></form></div></main>
<script>
let state,index=0;const $=id=>document.getElementById(id);
function draw(nodes){const c=$("pitch"),x=c.getContext("2d"),W=c.width,H=c.height;x.clearRect(0,0,W,H);x.strokeStyle="#d7ead9";x.lineWidth=2;
x.strokeRect(8,8,W-16,H-16);x.beginPath();x.moveTo(W/2,8);x.lineTo(W/2,H-8);x.stroke();x.beginPath();x.arc(W/2,H/2,70,0,Math.PI*2);x.stroke();
for(const n of nodes){x.beginPath();x.arc(n.x*W,n.y*H,n.role==="ball"?7:11,0,Math.PI*2);x.fillStyle=n.team==="possession"?"#61a8ff":n.team==="pressing"?"#ff6c62":"#fff06a";x.fill();if(n.controls_ball){x.strokeStyle="#fff";x.lineWidth=3;x.stroke()}}}
function show(i){index=Math.max(0,Math.min(i,state.items.length-1));const r=state.items[index],a=state.annotations[r.review_id]||{};$("frame").src="/images/"+r.image;draw(r.nodes);
$("meta").textContent=`${r.sequence} · frame ${r.frame} · ${r.visible_nodes} visible nodes`;for(const id of ["build_up","press_present","orientation_valid","graph_sufficient"])$(id).checked=a[id]??(id.endsWith("valid")||id==="graph_sufficient");
$("situation").value=a.situation||"ambiguous";$("confidence").value=a.confidence||"medium";$("rationale").value=a.rationale||"";$("progress").textContent=`${state.reviewed}/${state.items.length} reviewed · ${index+1}`;}
async function save(e){e.preventDefault();const r=state.items[index],body={review_id:r.review_id,situation:$("situation").value,confidence:$("confidence").value,rationale:$("rationale").value};
for(const id of ["build_up","press_present","orientation_valid","graph_sufficient"])body[id]=$(id).checked;const q=await fetch("/api/annotation",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});if(!q.ok){alert(await q.text());return}if(!state.annotations[r.review_id])state.reviewed++;state.annotations[r.review_id]=await q.json();show(index+1)}
$("form").onsubmit=save;$("prev").onclick=()=>show(index-1);fetch("/api/state").then(r=>r.json()).then(s=>{state=s;for(const v of s.situations)$("situation").add(new Option(v,v));show(Math.max(0,s.items.findIndex(x=>!s.annotations[x.review_id])))});
</script></body></html>"""


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], directory: Path) -> None:
        self.directory = directory.resolve()
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.annotation_path = directory / "annotations.json"
        super().__init__(address, Handler)

    def annotations(self) -> dict:
        return json.loads(self.annotation_path.read_text()) if self.annotation_path.exists() else {}


class Handler(BaseHTTPRequestHandler):
    server: Server

    def send(self, body: bytes, kind: str, status: int = 200) -> None:
        self.send_response(status);self.send_header("Content-Type", kind);self.send_header("Content-Length", str(len(body)));self.end_headers();self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/": self.send(HTML.encode(), "text/html; charset=utf-8");return
        if route == "/api/state":
            annotations=self.server.annotations();payload={**self.server.manifest,"annotations":annotations,"reviewed":len(annotations)}
            self.send(json.dumps(payload).encode(),"application/json");return
        if route.startswith("/images/"):
            path=(self.server.directory/unquote(route[8:])).resolve()
            if path.parent!=self.server.directory or not path.is_file(): self.send_error(HTTPStatus.NOT_FOUND);return
            self.send(path.read_bytes(),"image/jpeg");return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path!="/api/annotation": self.send_error(HTTPStatus.NOT_FOUND);return
        try:
            row=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))))
            valid={x["review_id"] for x in self.server.manifest["items"]}
            if row.get("review_id") not in valid: raise ValueError("Unknown review id")
            if row.get("situation") not in self.server.manifest["situations"]: raise ValueError("Invalid situation")
            row["updated_at"]=datetime.now(timezone.utc).isoformat();annotations=self.server.annotations();annotations[row["review_id"]]=row
            self.server.annotation_path.write_text(json.dumps(annotations,indent=2)+"\n");self.send(json.dumps(row).encode(),"application/json")
        except (ValueError,json.JSONDecodeError) as error:self.send(str(error).encode(),"text/plain",400)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--directory",type=Path,default=Path("data/review/gsr_expert"));parser.add_argument("--host",default="127.0.0.1");parser.add_argument("--port",type=int,default=8766);args=parser.parse_args()
    server=Server((args.host,args.port),args.directory);print(f"Expert review UI: http://{args.host}:{args.port}");server.serve_forever()


if __name__=="__main__":main()
