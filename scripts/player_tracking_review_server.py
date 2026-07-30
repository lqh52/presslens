#!/usr/bin/env python3
"""Serve a local visual comparison of player-tracking benchmark results."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PressLens · Player tracking review</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#07110d;color:#eef7f1}
*{box-sizing:border-box}body{margin:0}.app{width:min(1440px,100%);margin:auto;padding:24px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:17px;flex-wrap:wrap}
.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:10px;color:#789184;font-weight:800}
h1{margin:4px 0;font-size:25px}.muted{color:#8da096;font-size:12px;line-height:1.5}
.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(330px,.72fr);gap:17px}
.panel{background:#101b15;border:1px solid #26392f;border-radius:16px;padding:15px;min-width:0}
.stage{position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:11px;overflow:hidden}
.pitch-stage{position:relative;width:100%;aspect-ratio:105/68;background:#263f25;border-radius:11px;overflow:hidden;margin-top:12px}
video,canvas{position:absolute;inset:0;width:100%;height:100%}canvas{pointer-events:none}
.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}
button,.select{border:1px solid #30483b;background:#17261e;color:#eef7f1;border-radius:9px;padding:9px 11px;font:inherit;font-weight:700}
button{cursor:pointer}.model{background:#b8f36b;color:#10200e;border-color:#b8f36b}.model.alt{background:#1a2b22;color:#dce9e1;border-color:#385345}
button:disabled{opacity:.35;cursor:not-allowed}
.toggle{display:flex;align-items:center;gap:6px;font-size:12px;color:#a9bab1;margin-left:auto}
.frame{font-variant-numeric:tabular-nums;color:#a9bab1;font-size:12px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}
.metric{background:#0a130e;border:1px solid #24372d;border-radius:11px;padding:11px}
.metric strong{font-size:22px;display:block;margin:3px 0}.metric span{font-size:10px;color:#81958a}
.verdict{border-left:3px solid #b8f36b;padding:9px 12px;background:#13231a;border-radius:0 9px 9px 0;line-height:1.5;font-size:13px}
.tactic{margin:0 0 12px;padding:12px;border:1px solid #5c6031;background:#211f10;border-radius:11px}.tactic strong{display:block;font-size:19px;color:#fff06a;margin:3px 0}.tactic-detail{font-size:11px;color:#c9c69d;line-height:1.5}
textarea{width:100%;min-height:92px;resize:vertical;background:#09120d;border:1px solid #30483b;color:#eef7f1;border-radius:9px;padding:10px;font:inherit}
.save{width:100%;margin-top:8px;background:#b8f36b;color:#10200e}.saved{height:18px;color:#9fd25f;font-size:11px;margin-top:6px}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:9px;font-size:11px;color:#8da096}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.tuner{margin:0 0 13px;padding:12px;background:#0a130e;border:1px solid #24372d;border-radius:11px}.tuner-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:8px}.tuner-grid{display:grid;grid-template-columns:90px 1fr 42px;gap:7px 9px;align-items:center;font-size:11px;color:#a9bab1}.tuner input{width:100%;accent-color:#b8f36b}.tuner output{text-align:right;font-variant-numeric:tabular-nums;color:#eef7f1}.anchor-note{font-size:10px;color:#789184;line-height:1.45;margin-top:9px}.reset{padding:5px 8px;font-size:10px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.app{padding:13px}.toggle{margin-left:0}}
</style></head><body><main class="app">
<header><div><div class="eyebrow">Published evidence · local analysis</div><h1>Player tracking review</h1>
<div class="muted" id="clip-subtitle">Published PressLens evidence · 25 fps</div></div>
<div><a class="select" href="/tactics">Label tactics</a> <a class="select" href="/label">Label track objects</a><div class="muted" id="runtime"></div></div></header>
<div class="grid"><section class="panel">
 <div class="stage"><video id="video" controls playsinline preload="auto"></video><canvas id="overlay"></canvas></div>
 <div class="pitch-stage"><canvas id="pitch"></canvas></div>
 <div class="toolbar"><select id="tactic-filter" class="select"><option value="all">All tactics</option><option value="traps">Trap left + right</option><option value="trap_left">Trap left</option><option value="trap_right">Trap right</option><option value="high_press">High press</option><option value="central_screen">Central screen</option><option value="unstructured">Unstructured</option></select><select id="clip-select" class="select"></select><button id="baseline" class="model alt">YOLO11m baseline</button><button id="candidate" class="model">YOLO26m · high recall</button>
 <span class="frame" id="frame">Frame 0 / 99</span>
 <label class="toggle"><input id="identity" type="checkbox" checked> Identity</label>
 <label class="toggle"><input id="graph-edges" type="checkbox" checked> Graph edges</label>
 <label class="toggle"><input id="ball" type="checkbox" checked> Ball</label>
 <label class="toggle"><input id="ids" type="checkbox" checked> Track IDs</label>
 <label class="toggle"><input id="confidence" type="checkbox"> Confidence</label></div>
 <div class="legend"><span><i class="dot" style="background:#59a7ff"></i>Team A</span><span><i class="dot" style="background:#ff6868"></i>Team B</span><span><i class="dot" style="background:#ffd166"></i>Other</span><span>◇ goalkeeper candidate</span></div>
</section><aside class="panel">
 <div class="eyebrow" id="model-name"></div>
 <div class="tactic"><div class="eyebrow">Tactical pattern · synthetic graph model</div><strong id="tactic-name">Insufficient evidence</strong><div class="tactic-detail" id="tactic-detail">Waiting for projected ball and players…</div></div>
 <div class="tuner">
  <div class="tuner-head"><span class="eyebrow">Match anchor tuning</span><button class="reset" id="reset-tuning">Reset</button></div>
  <div class="tuner-grid">
   <label for="dino-weight">DINOv2</label><input id="dino-weight" type="range" min="0" max="100" value="45"><output id="dino-value">45%</output>
   <label for="reid-weight">PRTReID</label><input id="reid-weight" type="range" min="0" max="100" value="35"><output id="reid-value">35%</output>
   <label for="color-weight">Jersey colour</label><input id="color-weight" type="range" min="0" max="100" value="20"><output id="color-value">20%</output>
   <label for="validation">Other rescue</label><input id="validation" type="range" min="0" max="100" value="75"><output id="validation-value">75%</output>
  </div>
  <div class="anchor-note" id="anchor-note"></div>
  <button class="save" id="submit-tuning">Submit weights</button>
  <div class="saved" id="tuning-saved"></div>
 </div>
 <div class="metrics">
  <div class="metric"><span>Players / frame</span><strong id="players"></strong><span id="players-note"></span></div>
  <div class="metric"><span>Adjacent ID retention</span><strong id="retention"></strong><span>higher is steadier</span></div>
  <div class="metric"><span>Plausible frames</span><strong id="plausible"></strong><span>6–22 detected people</span></div>
  <div class="metric"><span>Unique tracks</span><strong id="tracks"></strong><span id="track-note"></span></div>
 </div>
 <div class="verdict" id="verdict"></div>
 <p class="muted">This is an unlabelled operational comparison. “More players” improves recall only if the added boxes are real athletes; use the overlay to inspect spectators, staff, referees and duplicate boxes.</p>
 <label class="eyebrow" for="notes">Review notes</label><textarea id="notes" placeholder="False positives, missing players, ID switches, projection concerns…"></textarea>
 <button class="save" id="save">Save local review</button><div class="saved" id="saved"></div>
</aside></div></main>
<script>
"use strict";
const $=id=>document.getElementById(id);let state,active,activeClip,previousIds=new Set(),tuned={};
const pct=x=>`${(Number(x)*100).toFixed(1)}%`;
function clip(){return state.clips.find(x=>x.id===activeClip)}
function experiment(){return clip().experiments[active]}
function hasTactic(x,filter){const labels=filter==="traps"?new Set(["trap_left","trap_right"]):new Set([filter]);return filter==="all"||x.tactical_model?.frames?.some(row=>labels.has(row.prediction?.label))}
function populateClipSelect(){const filter=$("tactic-filter").value,clips=state.clips.filter(x=>hasTactic(x,filter));$("clip-select").innerHTML="";for(const x of clips)$("clip-select").add(new Option(x.title,x.id));if(!clips.length)return;selectClip(clips.some(x=>x.id===activeClip)?activeClip:clips[0].id)}
function select(name){if(!clip().experiments[name])return;active=name;previousIds=new Set();$("baseline").classList.toggle("alt",name!=="yolo11m-botsort-baseline");$("candidate").classList.toggle("alt",name!=="yolo26m-botsort-high-recall");renderMetrics();draw()}
function selectClip(id){activeClip=id;previousIds=new Set();const x=clip(),saved=state.review.tuning?.[x.fixture_id],filter=$("tactic-filter").value,labels=filter==="traps"?new Set(["trap_left","trap_right"]):new Set([filter]),first=filter==="all"?-1:x.tactical_model?.frames?.findIndex(row=>labels.has(row.prediction?.label)),video=$("video");if(saved){$("dino-weight").value=saved.dino;$("reid-weight").value=saved.prtreid;$("color-weight").value=saved.color;$("validation").value=saved.validation}video.onloadedmetadata=()=>{if(first>=0){video.currentTime=(first+.1)/state.fps;video.pause()}};video.onseeked=()=>{previousIds=new Set();draw();if(first>=0)renderTactic(first)};video.src=`/video?clip=${encodeURIComponent(id)}`;video.load();$("clip-subtitle").textContent=`${x.title} · ${x.frames} frames at ${state.fps} fps`;
 $("baseline").disabled=!x.experiments["yolo11m-botsort-baseline"];$("candidate").disabled=!x.experiments["yolo26m-botsort-high-recall"];
const preferred=x.experiments["yolo26m-botsort-high-recall"]?"yolo26m-botsort-high-recall":Object.keys(x.experiments)[0];select(preferred);updateTuning()}
function weights(){const raw=[$("dino-weight").valueAsNumber,$("reid-weight").valueAsNumber,$("color-weight").valueAsNumber],sum=raw.reduce((a,b)=>a+b,0)||1;return{dino:raw[0]/sum,prtreid:raw[1]/sum,color:raw[2]/sum}}
function updateTuning(){for(const name of ["dino","reid","color","validation"])$(`${name}-value`).textContent=`${$(`${name}${name==="validation"?"":"-weight"}`).value}%`;reclassify();renderMetrics();draw()}
const sqdist=(a,b)=>a.reduce((sum,value,index)=>sum+(value-b[index])**2,0);
const unit=a=>{const norm=Math.sqrt(a.reduce((sum,value)=>sum+value*value,0))||1;return a.map(value=>value/norm)};
function reclassify(){if(!state)return;const w=weights(),strength=$("validation").valueAsNumber/100,fixture=clip()?.fixture_id;if(!fixture)return;
 const related=state.clips.filter(x=>x.fixture_id===fixture),all=[];
 for(const x of related)for(const original of Object.values(x.identities||{})){if(!original.tuning_features)continue;const threshold=x.identity_configuration?.role_threshold??.55,marginThreshold=x.identity_configuration?.team_margin_threshold??.08,manual=state.track_labels?.[`${x.id}:${original.track_id}`]?.label;all.push({clip:x,item:{...original},threshold,marginThreshold,manual})}
 const eligible=all.map((x,index)=>[x,index]).filter(([x])=>x.manual!=="other"&&(x.manual||x.item.role_probabilities.outfield>=x.threshold)&&x.item.detections>=3).map(([,index])=>index);if(eligible.length<2)return;
 const manualCluster=label=>label==="team_a"||label==="team_a_goalkeeper"?0:label==="team_b"||label==="team_b_goalkeeper"?1:null;
 let labels=eligible.map(index=>manualCluster(all[index].manual)??all[index].item.team_cluster??(index%2)),centers;
 for(let iteration=0;iteration<12;iteration++){centers={};for(const signal of ["dino","prtreid","color"]){centers[signal]=[0,1].map(cluster=>{const members=eligible.filter((_,position)=>labels[position]===cluster),dimensions=all[eligible[0]].item.tuning_features[signal].length,totalWeight=members.reduce((sum,index)=>sum+Math.min(all[index].item.detections,25),0)||1,center=Array(dimensions).fill(0);for(const index of members){const weight=Math.min(all[index].item.detections,25),vector=all[index].item.tuning_features[signal];for(let d=0;d<dimensions;d++)center[d]+=weight*vector[d]/totalWeight}return unit(center)})}
  const next=eligible.map(index=>{const fixed=manualCluster(all[index].manual);if(fixed!=null)return fixed;const feature=all[index].item.tuning_features,score=[0,1].map(cluster=>w.dino*sqdist(feature.dino,centers.dino[cluster])+w.prtreid*sqdist(feature.prtreid,centers.prtreid[cluster])+w.color*sqdist(feature.color,centers.color[cluster]));return score[0]<=score[1]?0:1});if(next.every((value,index)=>value===labels[index]))break;if(new Set(next).size===2)labels=next}
 for(const x of all){const feature=x.item.tuning_features,score=[0,1].map(cluster=>w.dino*sqdist(feature.dino,centers.dino[cluster])+w.prtreid*sqdist(feature.prtreid,centers.prtreid[cluster])+w.color*sqdist(feature.color,centers.color[cluster])),nearest=score[0]<=score[1]?0:1,second=1-nearest;
  x.item.team_cluster=nearest;x.item.team_margin=(score[second]-score[nearest])/Math.max(score[second],1e-9);const role=x.item.role_prediction,conf=x.item.role_confidence;x.item.goalkeeper=role==="goalkeeper"&&conf>=x.threshold;x.item.label=((role==="outfield"||x.item.goalkeeper)&&conf>=x.threshold&&(x.item.goalkeeper||x.item.team_margin>=x.marginThreshold))?`team_${nearest===0?"a":"b"}`:"other";x.item.postprocess_relabelled=false}
 for(const x of all){if(!x.manual)continue;x.item.goalkeeper=x.manual.endsWith("_goalkeeper");x.item.label=x.manual==="other"?"other":x.manual.startsWith("team_a")?"team_a":"team_b";x.item.team_cluster=x.item.label==="team_a"?0:x.item.label==="team_b"?1:null;x.item.manual_label=true}
 const total=all.reduce((n,x)=>n+x.item.detections,0),target=.35-.23*strength;let other=all.filter(x=>x.item.label==="other").reduce((n,x)=>n+x.item.detections,0);
 const candidates=all.filter(x=>!x.manual&&x.item.label==="other"&&x.item.prt_role_vote==="player"&&x.item.role_probabilities.outfield>=x.threshold*(1-.55*strength)&&x.item.team_margin>=x.marginThreshold*(1-.75*strength)).sort((a,b)=>(b.item.team_margin*b.item.role_probabilities.outfield*Math.min(b.item.detections,25))-(a.item.team_margin*a.item.role_probabilities.outfield*Math.min(a.item.detections,25)));
 for(const x of candidates){if(!total||other/total<=target)break;x.item.label=`team_${x.item.team_cluster===0?"a":"b"}`;x.item.postprocess_relabelled=true;other-=x.item.detections}
 tuned={};for(const x of related)tuned[x.id]={};for(const x of all)tuned[x.clip.id][String(x.item.track_id)]=x.item;
 const support=[0,1].map(cluster=>eligible.filter((_,position)=>labels[position]===cluster).length),rescued=all.filter(x=>x.item.postprocess_relabelled).length,changed=all.filter(x=>x.item.label!==(x.clip.identities?.[String(x.item.track_id)]?.label)).length,counts=all.reduce((a,x)=>(a[x.item.label]=(a[x.item.label]||0)+1,a),{});$("anchor-note").textContent=`${fixture}: re-clustered anchors ${support[0]} / ${support[1]} tracks · ${counts.team_a||0} A / ${counts.team_b||0} B / ${counts.other||0} Other · ${changed} assignments changed · ${rescued} Others rescued.`}
function renderMetrics(){const x=experiment(),m=x.metrics;$("model-name").textContent=x.label;$("players").textContent=Number(m.mean_players_per_frame).toFixed(2);
$("players-note").textContent=`median ${Number(m.median_players_per_frame).toFixed(0)}`;$("retention").textContent=pct(m.adjacent_id_retention);
$("plausible").textContent=pct(m.plausible_count_rate);$("tracks").textContent=m.unique_tracks;$("track-note").textContent=`median life ${Number(m.median_track_length).toFixed(0)} frames`;
$("runtime").textContent=`${x.label} · ${Number(m.runtime_seconds).toFixed(1)}s CPU`;
$("verdict").textContent=nameVerdict(active);$("notes").value=state.review.notes?.[activeClip]?.[active]||""}
function nameVerdict(name){const m=experiment().metrics,base=clip().experiments["yolo11m-botsort-baseline"]?.metrics,values=Object.values(tuned[activeClip]||clip().identities||{}),c=values.reduce((a,x)=>(a[x.label]=(a[x.label]||0)+1,a),{}),identity=` Tuned identity: ${c.team_a||0} Team A tracks, ${c.team_b||0} Team B, ${c.other||0} Other; ${values.filter(x=>x.goalkeeper).length} tentative goalkeeper signals.`;
if(name.includes("26m")&&base)return`Candidate result: ${Number(m.mean_players_per_frame).toFixed(2)} people/frame versus ${Number(base.mean_players_per_frame).toFixed(2)} in the baseline; adjacent ID retention is ${pct(m.adjacent_id_retention)} versus ${pct(base.adjacent_id_retention)}.${identity}`;
return`${experiment().label}: ${Number(m.mean_players_per_frame).toFixed(2)} people/frame with ${pct(m.adjacent_id_retention)} adjacent ID retention.${identity}`}
function renderTactic(index){const model=clip().tactical_model,result=model?.frames?.[index]?.prediction,summary=model?.summary;if(!result||result.label==="abstain"){$("tactic-name").textContent="Insufficient evidence";const reasons=result?.abstain_reasons?.join(", ")||"no reconstructed graph";$("tactic-detail").textContent=`${reasons} · clip accepted coverage ${pct(summary?.coverage||0)} · raw ${result?.raw_display||"—"}`;return}$("tactic-name").textContent=`${result.display} · ${Math.round((result.confidence||0)*100)}%`;$("tactic-detail").textContent=`${(result.possession_team||"").replace("_"," ").toUpperCase()} likely possession · ${result.visible_nodes||0} visible graph nodes · possession ${pct(result.possession_confidence||0)} · direction ${pct(result.direction_confidence||0)}`}
function colour(id,isNew){if(isNew)return"#ffcc66";const hue=(Number(id||0)*47)%360;return`hsl(${hue} 78% 64%)`}
function identityFor(d){return active==="yolo26m-botsort-high-recall"?(tuned[activeClip]?.[String(d.track_id)]||clip().identities?.[String(d.track_id)]):undefined}
function drawCanonicalPitch(index){const canvas=$("pitch"),ctx=canvas.getContext("2d"),w=1050,h=680,pad=28;canvas.width=w;canvas.height=h;ctx.fillStyle="#294a2a";ctx.fillRect(0,0,w,h);ctx.strokeStyle="rgba(238,247,241,.78)";ctx.lineWidth=3;ctx.strokeRect(pad,pad,w-2*pad,h-2*pad);ctx.beginPath();ctx.moveTo(w/2,pad);ctx.lineTo(w/2,h-pad);ctx.stroke();ctx.beginPath();ctx.arc(w/2,h/2,91.5,0,Math.PI*2);ctx.stroke();ctx.strokeRect(pad,h/2-201.5,157,403);ctx.strokeRect(w-pad-157,h/2-201.5,157,403);ctx.strokeRect(pad,h/2-91.5,52.5,183);ctx.strokeRect(w-pad-52.5,h/2-91.5,52.5,183);
 const projection=clip().projection?.frames?.[index],toCanvas=(x,y)=>[pad+(x+52.5)/105*(w-2*pad),pad+(y+34)/68*(h-2*pad)],nodes=(projection?.objects||[]).map(object=>({object,identity:tuned[activeClip]?.[String(object.track_id)]||clip().identities?.[String(object.track_id)],point:toCanvas(object.x,object.y),pitch:[object.x,object.y]})).filter(x=>x.identity?.label==="team_a"||x.identity?.label==="team_b"),edges=new Set();
 if($("graph-edges").checked){for(const team of ["team_a","team_b"]){const members=nodes.map((node,index)=>[node,index]).filter(([node])=>node.identity.label===team).map(([,index])=>index);if(members.length>1){const connected=[members[0]],remaining=new Set(members.slice(1));while(remaining.size){let best=null;for(const left of connected)for(const right of remaining){const distance=Math.hypot(nodes[left].pitch[0]-nodes[right].pitch[0],nodes[left].pitch[1]-nodes[right].pitch[1]);if(!best||distance<best.distance)best={left,right,distance}}edges.add([Math.min(best.left,best.right),Math.max(best.left,best.right)].join(":"));connected.push(best.right);remaining.delete(best.right)}}}
  for(let index=0;index<nodes.length;index++){const node=nodes[index],neighbours=nodes.map((other,otherIndex)=>({otherIndex,distance:otherIndex===index||other.identity.label!==node.identity.label?Infinity:Math.hypot(node.pitch[0]-other.pitch[0],node.pitch[1]-other.pitch[1])})).sort((a,b)=>a.distance-b.distance).slice(0,2);for(const neighbour of neighbours)if(neighbour.distance<=22)edges.add([Math.min(index,neighbour.otherIndex),Math.max(index,neighbour.otherIndex)].join(":"))}
  ctx.save();ctx.globalAlpha=.7;ctx.lineWidth=3;for(const edge of edges){const [left,right]=edge.split(":").map(Number),a=nodes[left],b=nodes[right];ctx.strokeStyle=a.identity.label==="team_a"?"#59a7ff":"#ff6868";ctx.beginPath();ctx.moveTo(...a.point);ctx.lineTo(...b.point);ctx.stroke()}
  ctx.strokeStyle="#ffd166";ctx.globalAlpha=.48;ctx.setLineDash([7,7]);for(let left=0;left<nodes.length;left++)for(let right=left+1;right<nodes.length;right++)if(nodes[left].identity.label!==nodes[right].identity.label&&Math.hypot(nodes[left].pitch[0]-nodes[right].pitch[0],nodes[left].pitch[1]-nodes[right].pitch[1])<=12){ctx.beginPath();ctx.moveTo(...nodes[left].point);ctx.lineTo(...nodes[right].point);ctx.stroke()}ctx.restore()}
 for(const node of nodes){const color=node.identity.label==="team_a"?"#59a7ff":"#ff6868";ctx.save();ctx.globalAlpha=node.object.estimated?.62:1;ctx.fillStyle=node.object.estimated?"#294a2a":color;ctx.strokeStyle=color;ctx.lineWidth=node.object.estimated?4:2;ctx.setLineDash(node.object.estimated?[4,3]:[]);ctx.beginPath();ctx.arc(node.point[0],node.point[1],node.identity.goalkeeper?11:8,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.restore();if($("ids").checked){ctx.fillStyle="#eef7f1";ctx.font="bold 15px system-ui";ctx.fillText(`#${node.object.track_id}`,node.point[0]+10,node.point[1]-9)}}const ball=clip().ball?.frames?.[index]?.ball;if($("ball").checked&&ball?.pitch_xy){const point=toCanvas(...ball.pitch_xy);ctx.save();ctx.strokeStyle="#fff06a";ctx.fillStyle=ball.method==="detected"?"#fff06a":"#294a2a";ctx.lineWidth=4;ctx.setLineDash(ball.method==="detected"?[]:[5,4]);ctx.beginPath();ctx.arc(point[0],point[1],7,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.restore()}const estimated=nodes.filter(node=>node.object.estimated).length;ctx.fillStyle="#eef7f1";ctx.font="bold 16px system-ui";ctx.fillText(`Canonical pitch · ${nodes.length} players · ${estimated} temporally estimated · ${Math.round((clip().projection?.coverage||0)*100)}% clip coverage`,pad+8,h-8)}
function drawBall(ctx,index){if(!$("ball").checked)return;const ball=clip().ball?.frames?.[index]?.ball;if(!ball)return;const [x,y]=ball.image_xy;ctx.save();ctx.strokeStyle="#fff06a";ctx.fillStyle=ball.method==="detected"?"rgba(255,240,106,.35)":"rgba(7,17,13,.45)";ctx.lineWidth=3;ctx.setLineDash(ball.method==="detected"?[]:[5,4]);ctx.beginPath();ctx.arc(x,y,10,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.beginPath();ctx.moveTo(x-14,y);ctx.lineTo(x+14,y);ctx.moveTo(x,y-14);ctx.lineTo(x,y+14);ctx.stroke();ctx.restore()}
function draw(){if(!state||!active)return;const v=$("video"),c=$("overlay"),ctx=c.getContext("2d"),w=v.videoWidth||1280,h=v.videoHeight||720;
if(c.width!==w||c.height!==h){c.width=w;c.height=h}ctx.clearRect(0,0,w,h);const index=Math.min(experiment().frames.length-1,Math.max(0,Math.floor(v.currentTime*state.fps)));
const detections=experiment().frames[index]?.detections||[],trackedDetections=detections.filter(d=>d.track_id!=null&&Number(d.confidence)>=.45),currentIds=new Set(trackedDetections.map(d=>d.track_id));drawCanonicalPitch(index);for(const d of trackedDetections){const [x1,y1,x2,y2]=d.bbox,isNew=!previousIds.has(d.track_id),identity=identityFor(d),identityColour=identity?.label==="team_a"?"#59a7ff":identity?.label==="team_b"?"#ff6868":"#ffd166",color=$("identity").checked&&identity?identityColour:colour(d.track_id,isNew);
ctx.strokeStyle=color;ctx.lineWidth=identity?.goalkeeper?5:3;ctx.setLineDash(identity?.goalkeeper?[8,5]:[]);ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);if($("ids").checked||$("confidence").checked||$("identity").checked){const bits=[];if($("identity").checked&&identity)bits.push(identity.label.replace("_"," ").toUpperCase()+(identity.goalkeeper?" · GK?":""));if($("ids").checked)bits.push(`#${d.track_id}`);if($("confidence").checked)bits.push(`${Math.round(d.confidence*100)}%`);
if(bits.length){ctx.font="bold 16px system-ui";const text=bits.join(" · "),tw=ctx.measureText(text).width;ctx.fillStyle="rgba(3,10,6,.82)";ctx.fillRect(x1,Math.max(0,y1-23),tw+12,23);ctx.fillStyle=color;ctx.fillText(text,x1+6,Math.max(17,y1-6))}}}drawBall(ctx,index);
previousIds=currentIds;renderTactic(index);$("frame").textContent=`Frame ${index} / ${experiment().frames.length-1} · ${trackedDetections.length} tracked people ≥45%`}
async function save(){const response=await fetch("/api/review",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({clip:activeClip,experiment:active,fixture:clip().fixture_id,notes:$("notes").value,tuning:{dino:$("dino-weight").valueAsNumber,prtreid:$("reid-weight").valueAsNumber,color:$("color-weight").valueAsNumber,validation:$("validation").valueAsNumber}})});
if(!response.ok){$("saved").textContent=await response.text();return}state.review=await response.json();$("saved").textContent="Saved locally"}
async function submitTuning(){await save();$("tuning-saved").textContent=$("saved").textContent==="Saved locally"?"Weights saved for this fixture":$("saved").textContent}
$("baseline").onclick=()=>select("yolo11m-botsort-baseline");$("candidate").onclick=()=>select("yolo26m-botsort-high-recall");$("save").onclick=save;
$("submit-tuning").onclick=submitTuning;
$("clip-select").onchange=event=>selectClip(event.target.value);
$("tactic-filter").onchange=populateClipSelect;
$("identity").onchange=draw;$("graph-edges").onchange=draw;$("ball").onchange=draw;$("ids").onchange=draw;$("confidence").onchange=draw;$("video").ontimeupdate=draw;
for(const id of ["dino-weight","reid-weight","color-weight","validation"])$(id).oninput=updateTuning;
$("reset-tuning").onclick=()=>{$("dino-weight").value=45;$("reid-weight").value=35;$("color-weight").value=20;$("validation").value=75;updateTuning()};
fetch("/api/state").then(r=>r.json()).then(s=>{state=s;const requested=new URLSearchParams(location.search).get("tactic");if([...$("tactic-filter").options].some(x=>x.value===requested))$("tactic-filter").value=requested;populateClipSelect()});
</script></body></html>"""

LABEL_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PressLens · Track labels</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#07110d;color:#eef7f1}
*{box-sizing:border-box}body{margin:0}.app{width:min(1500px,100%);margin:auto;padding:24px}
header{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:18px}
h1{font-size:25px;margin:0}.muted{color:#8da096;font-size:12px}.controls{display:flex;gap:8px;flex-wrap:wrap}
select,a,button{border:1px solid #30483b;background:#17261e;color:#eef7f1;border-radius:9px;padding:9px 11px;font:inherit;text-decoration:none}
.progress{margin:12px 0;color:#b8f36b;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:12px}
.card{background:#101b15;border:1px solid #26392f;border-radius:14px;overflow:hidden}.card.done{border-color:#789f55}
.crop{height:190px;width:100%;object-fit:contain;background:#020504}.body{padding:11px}.title{display:flex;justify-content:space-between;font-weight:800;margin-bottom:4px}
.suggestion{font-size:11px;color:#8da096;margin-bottom:9px}.choices{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.choices button{padding:7px;font-size:11px;cursor:pointer}.choices button.active{background:#b8f36b;color:#10200e;border-color:#b8f36b}
.choices .other{grid-column:span 2}.saved{position:fixed;right:18px;bottom:18px;background:#b8f36b;color:#10200e;padding:10px 14px;border-radius:9px;opacity:0;transition:.2s}.saved.show{opacity:1}
</style></head><body><main class="app">
<header><div><h1>Label tracked objects</h1><div class="muted">Labels become fixed match-level seeds; unlabelled tracks are inferred from them.</div></div>
<div class="controls"><select id="fixture"></select><select id="clip"></select><select id="status"><option value="all">All objects</option><option value="unlabelled">Unlabelled only</option><option value="labelled">Labelled only</option></select><a href="/">Back to review</a></div></header>
<div class="progress" id="progress"></div><section class="grid" id="grid"></section>
<div class="saved" id="saved">Label saved</div></main>
<script>
"use strict";const $=id=>document.getElementById(id);let state;
const names={team_a:"Team A",team_b:"Team B",other:"Other",team_a_goalkeeper:"GK · Team A",team_b_goalkeeper:"GK · Team B"};
function fixtures(){return[...new Set(state.tracks.map(x=>x.fixture_id))]}
function clips(){return[...new Set(state.tracks.filter(x=>x.fixture_id===$("fixture").value).map(x=>x.clip_id))]}
function populateClips(){const selected=$("clip").value;$("clip").innerHTML='<option value="all">All clips</option>';for(const value of clips())$("clip").add(new Option(value,value));if([...$("clip").options].some(x=>x.value===selected))$("clip").value=selected;render()}
function render(){const fixture=$("fixture").value,clip=$("clip").value,status=$("status").value,rows=state.tracks.filter(x=>x.fixture_id===fixture&&(clip==="all"||x.clip_id===clip)&&(status==="all"||status==="labelled"&&x.label||status==="unlabelled"&&!x.label));$("grid").innerHTML="";
 for(const row of rows){const card=document.createElement("article");card.className=`card ${row.label?"done":""}`;card.innerHTML=`<img class="crop" loading="lazy" src="${row.crop_url}"><div class="body"><div class="title"><span>#${row.track_id}</span><span>${row.clip_id.split("-").slice(-2).join(" ")}</span></div><div class="suggestion">Suggested: ${row.suggested_label?.replace("_"," ")||"unknown"} · ${Math.round((row.role_confidence||0)*100)}%</div><div class="choices">${Object.entries(names).map(([value,label])=>`<button data-value="${value}" class="${value===row.label?"active":""} ${value==="other"?"other":""}">${label}</button>`).join("")}</div></div>`;
  for(const button of card.querySelectorAll("button"))button.onclick=()=>save(row,button.dataset.value);$("grid").append(card)}
 const total=state.tracks.filter(x=>x.fixture_id===fixture).length,done=state.tracks.filter(x=>x.fixture_id===fixture&&x.label).length;$("progress").textContent=`${done} / ${total} objects labelled in this match · showing ${rows.length}`}
async function save(row,label){const response=await fetch("/api/track-label",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({clip_id:row.clip_id,track_id:row.track_id,label})});if(!response.ok){alert(await response.text());return}state.labels=await response.json();row.label=label;$("saved").classList.add("show");setTimeout(()=>$("saved").classList.remove("show"),900);render()}
$("fixture").onchange=populateClips;$("clip").onchange=render;$("status").onchange=render;
fetch("/api/label-state").then(r=>r.json()).then(s=>{state=s;for(const value of fixtures())$("fixture").add(new Option(value,value));populateClips()});
</script></body></html>"""

TACTIC_LABEL_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PressLens · Tactical labels</title><style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#07110d;color:#eef7f1}
*{box-sizing:border-box}body{margin:0}.app{width:min(1320px,100%);margin:auto;padding:24px}header{display:flex;justify-content:space-between;gap:16px;align-items:center;flex-wrap:wrap}
h1{margin:0 0 5px;font-size:26px}.muted{color:#91a399;font-size:12px;line-height:1.5}a,button,select{border:1px solid #30483b;background:#17261e;color:#eef7f1;border-radius:9px;padding:9px 12px;font:inherit;text-decoration:none}
.definitions{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:9px;margin:17px 0}.definition{border:1px solid #2a4034;background:#0d1812;border-radius:12px;padding:11px}.definition strong{color:#fff06a}.definition p{font-size:11px;color:#a7b6ae;line-height:1.45;margin:6px 0 0}
.layout{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(300px,.7fr);gap:16px}.panel{background:#101b15;border:1px solid #26392f;border-radius:15px;padding:15px}.stage{background:#000;aspect-ratio:16/9;border-radius:11px;overflow:hidden}video{width:100%;height:100%}
.proposal{font-size:12px;color:#8da096}.proposal h2{font-size:25px;color:#fff06a;margin:8px 0}.evidence{background:#09120d;border-radius:9px;padding:11px;line-height:1.6;margin:12px 0}.choices{display:grid;grid-template-columns:1fr 1fr;gap:9px}.choices button{font-size:18px;font-weight:800;cursor:pointer}.yes{border-color:#8fc856}.no{border-color:#d16a6a}.yes.active{background:#8fc856;color:#10200e}.no.active{background:#d16a6a;color:#190a0a}
.nav{display:flex;gap:8px;margin-top:12px}.nav button{flex:1}.progress{color:#b8f36b;margin:12px 0;font-size:12px}.done{opacity:.55}@media(max-width:850px){.layout{grid-template-columns:1fr}.app{padding:13px}}
</style></head><body><main class="app"><header><div><h1>Validate tactical proposals</h1><div class="muted">Synthetic geometry proposals · answer only Yes or No · judge the pattern around the displayed moment, not the entire match.</div></div><div><select id="filter"><option value="unlabelled">Unlabelled</option><option value="all">All</option></select> <a href="/">Back to review</a></div></header>
<section class="definitions" id="definitions"></section><div class="progress" id="progress"></div>
<div class="layout"><section class="panel"><div class="stage"><video id="video" controls playsinline preload="auto"></video></div><div class="nav"><button id="previous">← Previous</button><button id="replay">Replay moment</button><button id="next">Next →</button></div></section>
<aside class="panel proposal"><span id="position"></span><h2 id="pattern"></h2><div id="definition"></div><div class="evidence" id="evidence"></div><div class="choices"><button class="yes" id="yes">Yes</button><button class="no" id="no">No</button></div></aside></div>
<script>
"use strict";const $=id=>document.getElementById(id);let state,rows=[],index=0;
function filtered(){return state.candidates.filter(row=>$("filter").value==="all"||row.answer==null)}
function updateRows(preferred){rows=filtered();if(!rows.length){$("progress").textContent="All proposals labelled.";return}index=Math.max(0,Math.min(rows.length-1,preferred??index));render()}
function render(){const row=rows[index];if(!row)return;const v=$("video");v.src=`/video?clip=${encodeURIComponent(row.clip_id)}`;v.onloadedmetadata=()=>{v.currentTime=Math.max(0,row.frame/state.fps-.65);v.play().catch(()=>{})};$("position").textContent=`${row.clip_id} · frame ${row.frame} · proposal ${index+1}/${rows.length}`;$("pattern").textContent=`Is this ${row.display}?`;$("definition").textContent=state.definitions[row.label];const p=row.evidence.pressure||{};$("evidence").textContent=`Model confidence ${Math.round(row.evidence.confidence*100)}% · likely ${(row.evidence.possession_team||"unknown").replace("_"," ")} possession · ${p.within_12m??0} defenders within 12m · nearest defender ${p.nearest_defender_m??"—"}m · ${row.evidence.reason.replaceAll("_"," ")}`;$("yes").classList.toggle("active",row.answer===true);$("no").classList.toggle("active",row.answer===false);const done=state.candidates.filter(x=>x.answer!=null).length;$("progress").textContent=`${done} / ${state.candidates.length} proposals labelled · ${state.candidates.length-done} remaining`}
async function answer(value){const row=rows[index],response=await fetch("/api/tactic-label",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({candidate_id:row.id,answer:value})});if(!response.ok){alert(await response.text());return}row.answer=value;const original=state.candidates.find(x=>x.id===row.id);original.answer=value;updateRows($("filter").value==="unlabelled"?index:index+1)}
$("yes").onclick=()=>answer(true);$("no").onclick=()=>answer(false);$("previous").onclick=()=>{index=Math.max(0,index-1);render()};$("next").onclick=()=>{index=Math.min(rows.length-1,index+1);render()};$("replay").onclick=()=>{const row=rows[index],v=$("video");v.currentTime=Math.max(0,row.frame/state.fps-.65);v.play()};$("filter").onchange=()=>updateRows(0);
fetch("/api/tactic-label-state").then(r=>r.json()).then(s=>{state=s;$("definitions").innerHTML=Object.entries(s.definitions).map(([key,value])=>`<article class="definition"><strong>${s.names[key]}</strong><p>${value}</p></article>`).join("");updateRows(0)});
</script></main></body></html>"""

SYNTHETIC_LABEL_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PressLens · Synthetic graph labels</title><style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#07110d;color:#eef7f1}*{box-sizing:border-box}body{margin:0}.app{width:min(1280px,100%);margin:auto;padding:24px}
header{display:flex;justify-content:space-between;gap:15px;align-items:center;flex-wrap:wrap}h1{margin:0 0 5px;font-size:27px}.muted{font-size:12px;color:#90a299;line-height:1.5}a,button{border:1px solid #30483b;background:#17261e;color:#eef7f1;border-radius:9px;padding:9px 12px;font:inherit;text-decoration:none;cursor:pointer}
.layout{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(330px,.7fr);gap:16px;margin-top:16px}.panel{background:#101b15;border:1px solid #26392f;border-radius:15px;padding:15px}canvas{display:block;width:100%;aspect-ratio:105/68;background:#294a2a;border-radius:11px}
.legend{display:flex;gap:14px;font-size:11px;color:#98aaa0;margin-top:9px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:4px}.choices{display:grid;gap:7px;margin-top:12px}.choices button{text-align:left;font-weight:800}.choices button.active{background:#b8f36b;color:#10200e;border-color:#b8f36b}.definition{font-size:11px;color:#9fb0a7;line-height:1.45;margin-top:4px;font-weight:400}.progress{color:#b8f36b;font-size:12px;margin-top:10px}.nav{display:flex;gap:8px;margin-top:13px}.nav button{flex:1}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:10px;color:#789184;font-weight:800}
@media(max-width:850px){.layout{grid-template-columns:1fr}.app{padding:13px}}</style></head><body><main class="app">
<header><div><h1>Validate synthetic canonical graphs</h1><div class="muted">Confirm or reject the proposed tactical class. Yes creates a positive example; No creates a hard negative for that class.</div></div><a href="/">Back to real-video review</a></header>
<div class="layout"><section class="panel"><canvas id="pitch" width="1050" height="680"></canvas><div class="legend"><span><i class="dot" style="background:#59a7ff"></i>Possession</span><span><i class="dot" style="background:#ff6868"></i>Pressing</span><span><i class="dot" style="background:#fff06a"></i>Ball</span><span>Arrows = movement</span></div><div class="nav"><button id="previous">← Previous</button><button id="next">Next unlabelled →</button></div></section>
<aside class="panel"><div class="eyebrow" id="position"></div><h2 id="question"></h2><div class="definition" id="definition"></div><div class="choices"><button id="yes">Yes, correct</button><button id="no">No, it is another pattern</button></div><div class="choices" id="corrections" hidden></div><div class="progress" id="progress"></div></aside></div>
<script>
"use strict";const $=id=>document.getElementById(id);let state,index=0;
function draw(){const row=state.samples[index],c=$("pitch"),ctx=c.getContext("2d"),w=c.width,h=c.height,p=28,to=(x,y)=>[p+x*(w-2*p),p+y*(h-2*p)];ctx.fillStyle="#294a2a";ctx.fillRect(0,0,w,h);ctx.strokeStyle="rgba(238,247,241,.78)";ctx.lineWidth=3;ctx.strokeRect(p,p,w-2*p,h-2*p);ctx.beginPath();ctx.moveTo(w/2,p);ctx.lineTo(w/2,h-p);ctx.stroke();ctx.beginPath();ctx.arc(w/2,h/2,91.5,0,Math.PI*2);ctx.stroke();ctx.strokeRect(p,h/2-201.5,157,403);ctx.strokeRect(w-p-157,h/2-201.5,157,403);
 const players=row.nodes.filter(n=>n.team!=="ball");for(const team of ["possession","pressing"]){const members=players.filter(n=>n.team===team);ctx.save();ctx.strokeStyle=team==="possession"?"rgba(89,167,255,.38)":"rgba(255,104,104,.38)";ctx.lineWidth=2;for(const a of members){const nearest=members.filter(b=>b!==a).sort((b,d)=>Math.hypot(a.x-b.x,a.y-b.y)-Math.hypot(a.x-d.x,a.y-d.y)).slice(0,2);for(const b of nearest){ctx.beginPath();ctx.moveTo(...to(a.x,a.y));ctx.lineTo(...to(b.x,b.y));ctx.stroke()}}ctx.restore()}
 for(const n of row.nodes){const [x,y]=to(n.x,n.y);if(n.team==="ball"){ctx.fillStyle="#fff06a";ctx.strokeStyle="#342f05";ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.fill();ctx.stroke();continue}const color=n.team==="possession"?"#59a7ff":"#ff6868";ctx.fillStyle=color;ctx.strokeStyle=n.controls_ball?"#fff06a":"#eef7f1";ctx.lineWidth=n.controls_ball?5:2;ctx.beginPath();ctx.arc(x,y,n.role==="goalkeeper"?11:8,0,Math.PI*2);ctx.fill();ctx.stroke();const scale=420;ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+n.vx*scale,y+n.vy*scale);ctx.stroke()}
}
function render(){const row=state.samples[index];draw();$("position").textContent=`Synthetic graph ${index+1} / ${state.samples.length} · ${row.id}`;$("question").textContent=`Rule assigned: ${row.proposed_name}. Is this correct?`;$("definition").textContent=`Assignment rule: ${row.definition}`;$("yes").classList.toggle("active",row.answer===true);$("no").classList.toggle("active",row.answer===false);$("corrections").hidden=row.answer!==false;$("corrections").innerHTML=`<strong>If No, what is the correct pattern?</strong>`+state.classes.filter(x=>x.id!==row.proposed_label).map(x=>`<button data-label="${x.id}" class="${row.corrected_label===x.id?"active":""}">${x.name}<div class="definition">${x.definition}</div></button>`).join("")+`<button data-label="unsure" class="${row.corrected_label==="unsure"?"active":""}">Unsure / ambiguous</button>`;for(const b of $("corrections").querySelectorAll("button"))b.onclick=()=>save(false,b.dataset.label);const done=state.samples.filter(x=>x.answer!=null&&(x.answer||x.corrected_label)).length;$("progress").textContent=`${done} / ${state.samples.length} labelled · ${state.samples.length-done} remaining`}
async function save(answer,corrected_label=null){const row=state.samples[index],response=await fetch("/api/synthetic-tactic-label",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({sample_id:row.id,answer,corrected_label})});if(!response.ok){alert(await response.text());return}row.answer=answer;row.corrected_label=corrected_label;if(answer||corrected_label)nextUnlabelled();else render()}
function nextUnlabelled(){for(let step=1;step<=state.samples.length;step++){const candidate=(index+step)%state.samples.length,row=state.samples[candidate],complete=row.answer===true||(row.answer===false&&row.corrected_label);if(!complete){index=candidate;render();return}}render()}
$("yes").onclick=()=>save(true);$("no").onclick=()=>{state.samples[index].answer=false;state.samples[index].corrected_label=null;render()};$("previous").onclick=()=>{index=(index-1+state.samples.length)%state.samples.length;render()};$("next").onclick=nextUnlabelled;fetch("/api/synthetic-label-state").then(r=>r.json()).then(s=>{state=s;render()});
</script></main></body></html>"""

TRAP_EVENT_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PressLens · Trap events</title><style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#07110d;color:#eef7f1}*{box-sizing:border-box}body{margin:0}.app{width:min(1400px,100%);margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:center;gap:15px;flex-wrap:wrap}h1{margin:0}.muted{color:#92a39a;font-size:12px}a{color:#eef7f1;border:1px solid #30483b;border-radius:9px;padding:9px 12px;text-decoration:none}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:15px;margin-top:18px}.card{background:#101b15;border:1px solid #293d32;border-radius:15px;padding:14px}.head{display:flex;justify-content:space-between;gap:10px;margin-bottom:10px}.label{font-size:20px;font-weight:900;color:#fff06a}.detail{font-size:11px;color:#93a59b}video,canvas{display:block;width:100%;border-radius:10px;background:#000}canvas{aspect-ratio:105/68;margin-top:9px;background:#294a2a}@media(max-width:600px){.app{padding:12px}.grid{grid-template-columns:1fr}}
</style></head><body><main class="app"><header><div><h1>Trap event review</h1><div class="muted">Short 5 fps excerpts · exactly one fixed tactical label per video</div></div><a href="/?tactic=traps">Back to full clips</a></header><section class="grid" id="grid"></section></main>
<script>"use strict";const $=id=>document.getElementById(id);
function pitch(canvas,event){const ctx=canvas.getContext("2d"),w=1050,h=680,p=28,to=(x,y)=>[p+(x+52.5)/105*(w-2*p),p+(y+34)/68*(h-2*p)];canvas.width=w;canvas.height=h;ctx.fillStyle="#294a2a";ctx.fillRect(0,0,w,h);ctx.strokeStyle="#dce8df";ctx.lineWidth=3;ctx.strokeRect(p,p,w-2*p,h-2*p);ctx.beginPath();ctx.moveTo(w/2,p);ctx.lineTo(w/2,h-p);ctx.stroke();ctx.beginPath();ctx.arc(w/2,h/2,91.5,0,Math.PI*2);ctx.stroke();const identities=event.canonical.identities;for(const object of event.canonical.objects){const identity=identities[String(object.track_id)];if(!identity||!["team_a","team_b"].includes(identity.label))continue;const [x,y]=to(object.x,object.y),color=identity.label==="team_a"?"#59a7ff":"#ff6868";ctx.fillStyle=color;ctx.strokeStyle="#eef7f1";ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,identity.goalkeeper?11:8,0,Math.PI*2);ctx.fill();ctx.stroke()}const ball=event.canonical.ball;if(ball?.pitch_xy){const [x,y]=to(...ball.pitch_xy);ctx.fillStyle="#fff06a";ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.fill()}}
fetch("/api/trap-events").then(r=>r.json()).then(state=>{for(const event of state.events){const card=document.createElement("article");card.className="card";card.innerHTML=`<div class="head"><div><div class="label">${event.display}</div><div class="detail">${event.clip_id} · source frames ${event.source_frames[0]}–${event.source_frames[1]}</div></div><div class="detail">${Math.round(event.confidence*100)}% · ${event.fps} fps · ${event.duration_seconds}s</div></div><video controls loop muted playsinline preload="metadata" src="/trap-video?event=${encodeURIComponent(event.id)}"></video><canvas></canvas>`;$("grid").append(card);pitch(card.querySelector("canvas"),event)}});</script></body></html>"""

REVIEWED_EVENT_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PressLens · Reviewed tactical events</title><style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#07110d;color:#eef7f1}*{box-sizing:border-box}body{margin:0}.app{width:min(1500px,100%);margin:auto;padding:22px}header{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}h1{margin:0}.muted{font-size:12px;color:#91a399}a{color:#eef7f1;text-decoration:none;border:1px solid #30483b;border-radius:9px;padding:9px 12px}.grid{display:grid;gap:16px;margin-top:18px}.card{background:#101b15;border:1px solid #293e33;border-radius:15px;padding:14px}.head{display:flex;justify-content:space-between;gap:10px;margin-bottom:10px}.label{font-size:21px;font-weight:900;color:#fff06a}.detail{font-size:11px;color:#98aaa0}.videos{display:grid;grid-template-columns:1fr 1fr;gap:10px}.videos h3{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#8da096;margin:0 0 5px}.videos video{display:block;width:100%;border-radius:10px;background:#000}@media(max-width:850px){.videos{grid-template-columns:1fr}.app{padding:12px}}
</style></head><body><main class="app"><header><div><h1>Reviewed tactical event excerpts</h1><div class="muted">Original eight videos · 5 fps · one fixed label · synchronized broadcast and canonical graph</div></div><a href="/">Back to expansion review</a></header><section class="grid" id="grid"></section></main><script>
"use strict";const grid=document.getElementById("grid");fetch("/api/reviewed-events").then(r=>r.json()).then(state=>{for(const event of state.events){const card=document.createElement("article");card.className="card";card.innerHTML=`<div class="head"><div><div class="label">${event.display}</div><div class="detail">${event.clip_id} · source frames ${event.source_frames[0]}–${event.source_frames[1]}</div></div><div class="detail">${event.fps} fps · ${event.sampled_frames.length} sampled frames</div></div><div class="videos"><div><h3>Broadcast · boxes, IDs, team edges, ball</h3><video controls loop muted playsinline preload="metadata" src="/reviewed-event-video?event=${encodeURIComponent(event.id)}&kind=broadcast"></video></div><div><h3>Canonical graph · nodes, team edges, pressure, ball</h3><video controls loop muted playsinline preload="metadata" src="/reviewed-event-video?event=${encodeURIComponent(event.id)}&kind=canonical"></video></div></div>`;grid.append(card)}});</script></body></html>"""


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], directory: Path) -> None:
        self.directory = directory.resolve()
        self.review_path = self.directory / "review.json"
        self.track_labels_path = self.directory / "track-labels.json"
        self.tactic_labels_path = self.directory / "tactical-labels.json"
        self.synthetic_tactic_labels_path = (
            self.directory / "synthetic-tactic-labels.json"
        )
        trap_manifest = self.directory / "trap-events" / "manifest.json"
        self.trap_events = (
            json.loads(trap_manifest.read_text()).get("events", [])
            if trap_manifest.exists()
            else []
        )
        reviewed_manifest = self.directory / "reviewed-events" / "manifest.json"
        self.reviewed_events = (
            json.loads(reviewed_manifest.read_text()).get("events", [])
            if reviewed_manifest.exists()
            else []
        )
        synthetic_path = self.directory.parents[1] / "data" / "graphs" / "synthetic.npz"
        synthetic = np.load(synthetic_path)
        self.synthetic_features = synthetic["features"]
        self.synthetic_targets = synthetic["labels"]
        self.synthetic_label_names = synthetic["label_names"].tolist()
        self.synthetic_indices = [
            int(index)
            for label in range(len(self.synthetic_label_names))
            for index in np.flatnonzero(self.synthetic_targets == label)[:20]
        ]
        self.label_lock = threading.Lock()
        self.identities = {
            payload["clip_id"]: payload
            for path in sorted((self.directory / "identities").glob("*.json"))
            for payload in [json.loads(path.read_text())]
            if "clip_id" in payload
        }
        anchors_path = self.directory / "identities" / "match-anchors.json"
        self.anchors = json.loads(anchors_path.read_text()) if anchors_path.exists() else {}
        self.projections = {
            payload["clip_id"]: payload
            for path in sorted((self.directory / "pitch-projections").glob("*.json"))
            for payload in [json.loads(path.read_text())]
        }
        self.balls = {
            payload["clip_id"]: payload
            for path in sorted((self.directory / "ball-tracking").glob("*.json"))
            for payload in [json.loads(path.read_text())]
        }
        self.tactical_patterns = {
            payload["clip_id"]: payload
            for path in sorted((self.directory / "tactical-patterns").glob("*.json"))
            for payload in [json.loads(path.read_text())]
        }
        self.tactical_model = {
            payload["clip_id"]: payload
            for path in sorted((self.directory / "tactical-model").glob("*.json"))
            for payload in [json.loads(path.read_text())]
        }
        self.results: dict[str, dict[str, dict]] = {}
        for path in sorted((self.directory / "results").glob("*/*.json")):
            payload = json.loads(path.read_text())
            self.results.setdefault(payload["clip_id"], {})[
                payload["experiment"]["name"]
            ] = payload
        report = json.loads((self.directory / "report.json").read_text())
        self.metrics = {
            (clip["clip_id"], row["name"]): clip
            for row in report["experiments"]
            for clip in row["clips"]
        }
        self.videos = {
            clip_id: Path(
                experiments.get(
                    "yolo26m-botsort-high-recall",
                    next(iter(experiments.values())),
                )["clip_path"]
            ).resolve()
            for clip_id, experiments in self.results.items()
        }
        if not self.results or any(not path.is_file() for path in self.videos.values()):
            raise FileNotFoundError("Review directory needs published.mp4 and result JSON")
        self.track_samples: dict[tuple[str, int], dict] = {}
        for clip_id, experiments in self.results.items():
            result = experiments.get("yolo26m-botsort-high-recall")
            if result is None:
                continue
            for frame in result["frames"]:
                for detection in frame["detections"]:
                    if (
                        detection.get("track_id") is None
                        or float(detection.get("confidence", 0.0)) < 0.45
                    ):
                        continue
                    key = (clip_id, int(detection["track_id"]))
                    box = detection["bbox"]
                    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                    score = float(detection.get("confidence", 0.0)) * area**0.5
                    if score > self.track_samples.get(key, {}).get("score", -1):
                        self.track_samples[key] = {
                            "frame": int(frame["frame"]),
                            "bbox": box,
                            "score": score,
                        }
        super().__init__(address, Handler)

    def review(self) -> dict:
        if self.review_path.exists():
            return json.loads(self.review_path.read_text())
        return {"notes": {}}

    def track_labels(self) -> dict:
        if self.track_labels_path.exists():
            return json.loads(self.track_labels_path.read_text()).get("labels", {})
        return {}

    @staticmethod
    def tactic_definitions() -> dict[str, str]:
        return {
            "touchline_trap": "YES when the ball is near a touchline and at least two defenders jointly close space from the inside, restricting play along the line. NO for incidental crowding or one defender pressing alone.",
            "intense_pressure": "YES when multiple defenders are tightly concentrated around the ball carrier and immediate passing options. This is spatial pressure; do not require it to occur high up the pitch. NO when players are merely nearby in a settled shape.",
            "central_block": "YES when the defending team protects the central corridor around the ball with two or more screening defenders. NO when the ball is wide or the central density is accidental.",
            "compact_block": "YES when the visible defending unit maintains a clearly compressed width and depth as a coordinated block. NO when compactness is caused mainly by camera cropping or too few visible players.",
            "low_pressure": "YES when the possessing player has clear space and no defender is close enough to apply immediate pressure. NO if an opponent is actively closing down or blocks the next action.",
            "unstructured": "YES when no stable coordinated press or defensive block is visible—typically a transition, broken play, or scattered shape. NO when another defined pattern is clearly present.",
        }

    def tactic_labels(self) -> dict:
        if self.tactic_labels_path.exists():
            return json.loads(self.tactic_labels_path.read_text()).get("labels", {})
        return {}

    def tactic_candidates(self) -> list[dict]:
        labels = self.tactic_labels()
        candidates = []
        for clip_id, payload in sorted(self.tactical_patterns.items()):
            runs: list[list[dict]] = []
            current: list[dict] = []
            for row in payload.get("frames", []):
                pattern = row.get("pattern", {})
                usable = (
                    pattern.get("label") not in {None, "abstain"}
                    and float(pattern.get("confidence", 0.0)) >= 0.35
                )
                if (
                    usable
                    and current
                    and pattern["label"] == current[-1]["pattern"]["label"]
                    and int(row["frame"]) == int(current[-1]["frame"]) + 1
                ):
                    current.append(row)
                else:
                    if current:
                        runs.append(current)
                    current = [row] if usable else []
            if current:
                runs.append(current)
            for run_index, run in enumerate(runs):
                representative = max(
                    run, key=lambda row: float(row["pattern"]["confidence"])
                )
                pattern = representative["pattern"]
                candidate_id = (
                    f"{clip_id}:{pattern['label']}:{run[0]['frame']}-{run[-1]['frame']}"
                )
                candidates.append(
                    {
                        "id": candidate_id,
                        "clip_id": clip_id,
                        "frame": int(representative["frame"]),
                        "start_frame": int(run[0]["frame"]),
                        "end_frame": int(run[-1]["frame"]),
                        "label": pattern["label"],
                        "display": pattern.get(
                            "display", pattern["label"].replace("_", " ").title()
                        ),
                        "evidence": pattern,
                        "answer": labels.get(candidate_id, {}).get("answer"),
                    }
                )
        return candidates

    def tactic_label_state(self) -> dict:
        definitions = self.tactic_definitions()
        return {
            "fps": 25,
            "definitions": definitions,
            "names": {
                key: key.replace("_", " ").title() for key in definitions
            },
            "candidates": self.tactic_candidates(),
        }

    @staticmethod
    def synthetic_definitions() -> dict[str, str]:
        return {
            "unstructured": "Scattered or transition-like defending with no coordinated screen, trap, or dense press.",
            "central_screen": "Defenders form layered cover across the central passing corridor ahead of the ball.",
            "trap_left": "Ball is near the possession team’s left touchline; pressure closes the inside and limits escape.",
            "trap_right": "Ball is near the possession team’s right touchline; pressure closes the inside and limits escape.",
            "high_press": "Several opponents tightly surround and move toward the ball in an aggressive coordinated press.",
        }

    def synthetic_tactic_labels(self) -> dict:
        if self.synthetic_tactic_labels_path.exists():
            return json.loads(self.synthetic_tactic_labels_path.read_text()).get(
                "labels", {}
            )
        return {}

    def synthetic_label_state(self) -> dict:
        saved = self.synthetic_tactic_labels()
        definitions = self.synthetic_definitions()
        samples = []
        for source_index in self.synthetic_indices:
            features = self.synthetic_features[source_index]
            sample_id = f"synthetic:{source_index}"
            proposed_label = self.synthetic_label_names[
                int(self.synthetic_targets[source_index])
            ]
            nodes = []
            for node_index, feature in enumerate(features):
                team_index = int(np.argmax(feature[4:7]))
                role_index = int(np.argmax(feature[7:12]))
                nodes.append(
                    {
                        "id": node_index,
                        "x": round(float(feature[0]), 5),
                        "y": round(float(feature[1]), 5),
                        "vx": round(float(feature[2]), 5),
                        "vy": round(float(feature[3]), 5),
                        "team": (
                            "possession"
                            if team_index == 0
                            else "pressing"
                            if team_index == 1
                            else "ball"
                        ),
                        "role": (
                            ["goalkeeper", "player", "referee", "other", "ball"][
                                role_index
                            ]
                        ),
                        "controls_ball": bool(feature[12] > 0.5),
                    }
                )
            samples.append(
                {
                    "id": sample_id,
                    "nodes": nodes,
                    "proposed_label": proposed_label,
                    "proposed_name": proposed_label.replace("_", " ").title(),
                    "definition": definitions[proposed_label],
                    "answer": (
                        saved.get(sample_id, {}).get("answer")
                        if saved.get(sample_id, {}).get("answer") is True
                        or saved.get(sample_id, {}).get("corrected_label")
                        else None
                    ),
                    "corrected_label": saved.get(sample_id, {}).get(
                        "corrected_label"
                    ),
                }
            )
        return {
            "classes": [
                {
                    "id": label,
                    "name": label.replace("_", " ").title(),
                    "definition": definitions[label],
                }
                for label in self.synthetic_label_names
            ],
            "samples": samples,
        }

    def label_state(self) -> dict:
        labels = self.track_labels()
        rows = []
        for (clip_id, track_id), _sample in sorted(self.track_samples.items()):
            identity = next(
                (
                    row
                    for row in self.identities.get(clip_id, {}).get("tracks", [])
                    if int(row["track_id"]) == track_id
                ),
                {},
            )
            fixture_id = self.identities.get(clip_id, {}).get("fixture_id")
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "clip_id": clip_id,
                    "track_id": track_id,
                    "crop_url": f"/crop?clip={clip_id}&track={track_id}",
                    "suggested_label": identity.get("label"),
                    "role_confidence": identity.get("role_confidence"),
                    "label": labels.get(f"{clip_id}:{track_id}", {}).get("label"),
                }
            )
        return {"tracks": rows, "labels": labels}

    def crop(self, clip_id: str, track_id: int) -> bytes | None:
        sample = self.track_samples.get((clip_id, track_id))
        video = self.videos.get(clip_id)
        if sample is None or video is None:
            return None
        capture = cv2.VideoCapture(str(video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame"])
        ok, image = capture.read()
        capture.release()
        if not ok:
            return None
        height, width = image.shape[:2]
        left, top, right, bottom = sample["bbox"]
        pad_x = 0.08 * (right - left)
        pad_y = 0.05 * (bottom - top)
        x1, y1 = max(0, int(left - pad_x)), max(0, int(top - pad_y))
        x2, y2 = min(width, int(right + pad_x)), min(height, int(bottom + pad_y))
        crop = image[y1:y2, x1:x2]
        if not crop.size:
            return None
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return encoded.tobytes() if ok else None

    def state(self) -> dict:
        return {
            "fps": 25,
            "clips": [
                {
                    "id": clip_id,
                    "title": clip_id.replace("-published", "").replace("-", " "),
                    "frames": max(
                        len(payload["frames"]) for payload in experiments.values()
                    ),
                    "identity_counts": self.identities.get(clip_id, {}).get("counts", {}),
                    "fixture_id": self.identities.get(clip_id, {}).get("fixture_id"),
                    "identity_configuration": self.identities.get(clip_id, {}).get("configuration", {}),
                    "goalkeepers": self.identities.get(clip_id, {}).get("goalkeepers", 0),
                    "projection": self.projections.get(clip_id),
                    "ball": self.balls.get(clip_id),
                    "tactical": self.tactical_patterns.get(clip_id),
                    "tactical_model": self.tactical_model.get(clip_id),
                    "identities": {
                        str(track["track_id"]): track
                        for track in self.identities.get(clip_id, {}).get("tracks", [])
                    },
                    "experiments": {
                        name: {
                            "label": (
                                "YOLO26m + BoT-SORT · 1280px high recall"
                                if "high-recall" in name
                                else "YOLO26m + BoT-SORT · 960px"
                                if "26m" in name
                                else "YOLO11m + BoT-SORT · 640px"
                            ),
                            "metrics": self.metrics[(clip_id, name)],
                            "frames": payload["frames"],
                        }
                        for name, payload in experiments.items()
                    },
                }
                for clip_id, experiments in sorted(self.results.items())
            ],
            "anchors": self.anchors,
            "track_labels": self.track_labels(),
            "review": self.review(),
        }


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
        elif route == "/label":
            self.send_bytes(LABEL_HTML.encode(), "text/html; charset=utf-8")
        elif route == "/tactics":
            self.send_bytes(
                SYNTHETIC_LABEL_HTML.encode(), "text/html; charset=utf-8"
            )
        elif route == "/trap-review":
            self.send_bytes(TRAP_EVENT_HTML.encode(), "text/html; charset=utf-8")
        elif route == "/reviewed-events":
            self.send_bytes(REVIEWED_EVENT_HTML.encode(), "text/html; charset=utf-8")
        elif route == "/api/trap-events":
            self.send_bytes(
                json.dumps({"events": self.server.trap_events}).encode(),
                "application/json",
            )
        elif route == "/api/reviewed-events":
            self.send_bytes(
                json.dumps({"events": self.server.reviewed_events}).encode(),
                "application/json",
            )
        elif route == "/api/state":
            self.send_bytes(json.dumps(self.server.state()).encode(), "application/json")
        elif route == "/api/label-state":
            self.send_bytes(
                json.dumps(self.server.label_state()).encode(), "application/json"
            )
        elif route == "/api/tactic-label-state":
            self.send_bytes(
                json.dumps(self.server.tactic_label_state()).encode(),
                "application/json",
            )
        elif route == "/api/synthetic-label-state":
            self.send_bytes(
                json.dumps(self.server.synthetic_label_state()).encode(),
                "application/json",
            )
        elif route == "/crop":
            query = parse_qs(urlparse(self.path).query)
            try:
                clip_id = query["clip"][0]
                track_id = int(query["track"][0])
            except (KeyError, ValueError, IndexError):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            body = self.server.crop(clip_id, track_id)
            if body is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(body, "image/jpeg")
        elif route == "/video":
            query = urlparse(self.path).query
            clip_id = next(
                (
                    item.partition("=")[2]
                    for item in query.split("&")
                    if item.partition("=")[0] == "clip"
                ),
                "",
            )
            self.send_video(clip_id)
        elif route == "/trap-video":
            query = parse_qs(urlparse(self.path).query)
            self.send_trap_video(query.get("event", [""])[0])
        elif route == "/reviewed-event-video":
            query = parse_qs(urlparse(self.path).query)
            self.send_reviewed_event_video(
                query.get("event", [""])[0], query.get("kind", [""])[0]
            )
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def send_video(self, clip_id: str) -> None:
        video = self.server.videos.get(clip_id)
        if video is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = video.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        status = HTTPStatus.OK
        if range_header and range_header.startswith("bytes="):
            raw_start, _, raw_end = range_header[6:].partition("-")
            start = int(raw_start) if raw_start else 0
            end = min(int(raw_end), end) if raw_end else end
            status = HTTPStatus.PARTIAL_CONTENT
        with video.open("rb") as handle:
            handle.seek(start)
            body = handle.read(end - start + 1)
        headers = {"Accept-Ranges": "bytes"}
        if status == HTTPStatus.PARTIAL_CONTENT:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_bytes(body, mimetypes.guess_type(video.name)[0] or "video/mp4", status, extra_headers=headers)

    def send_trap_video(self, event_id: str) -> None:
        event = next(
            (row for row in self.server.trap_events if row["id"] == event_id),
            None,
        )
        if event is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        video = (self.server.directory / "trap-events" / event["video"]).resolve()
        expected = (self.server.directory / "trap-events").resolve()
        if video.parent != expected or not video.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = video.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw_start, _, raw_end = range_header[6:].partition("-")
            start = int(raw_start) if raw_start else 0
            end = min(int(raw_end), end) if raw_end else end
            status = HTTPStatus.PARTIAL_CONTENT
        with video.open("rb") as handle:
            handle.seek(start)
            body = handle.read(end - start + 1)
        headers = {"Accept-Ranges": "bytes"}
        if status == HTTPStatus.PARTIAL_CONTENT:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_bytes(body, "video/mp4", status, extra_headers=headers)

    def send_reviewed_event_video(self, event_id: str, kind: str) -> None:
        event = next(
            (row for row in self.server.reviewed_events if row["id"] == event_id),
            None,
        )
        key = f"{kind}_video"
        if event is None or key not in {"broadcast_video", "canonical_video"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        root = (self.server.directory / "reviewed-events").resolve()
        video = (root / event[key]).resolve()
        if video.parent != root or not video.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = video.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw_start, _, raw_end = range_header[6:].partition("-")
            start = int(raw_start) if raw_start else 0
            end = min(int(raw_end), end) if raw_end else end
            status = HTTPStatus.PARTIAL_CONTENT
        with video.open("rb") as handle:
            handle.seek(start)
            body = handle.read(end - start + 1)
        headers = {"Accept-Ranges": "bytes"}
        if status == HTTPStatus.PARTIAL_CONTENT:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_bytes(body, "video/mp4", status, extra_headers=headers)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/track-label":
            self.save_track_label()
            return
        if route == "/api/tactic-label":
            self.save_tactic_label()
            return
        if route == "/api/synthetic-tactic-label":
            self.save_synthetic_tactic_label()
            return
        if route != "/api/review":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            experiment = str(payload["experiment"])
            clip_id = str(payload["clip"])
            if experiment not in self.server.results.get(clip_id, {}):
                raise ValueError("Unknown experiment")
            review = self.server.review()
            review.setdefault("notes", {}).setdefault(clip_id, {})[experiment] = str(
                payload.get("notes", "")
            )
            tuning = payload.get("tuning")
            if tuning is not None:
                values = {
                    key: float(tuning[key])
                    for key in ("dino", "prtreid", "color", "validation")
                }
                if any(value < 0 or value > 100 for value in values.values()):
                    raise ValueError("Tuning values must be between 0 and 100")
                fixture_id = str(payload["fixture"])
                if fixture_id != self.server.identities.get(clip_id, {}).get("fixture_id"):
                    raise ValueError("Fixture does not match clip")
                review.setdefault("tuning", {})[fixture_id] = values
            review["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.server.review_path.write_text(json.dumps(review, indent=2) + "\n")
            self.send_bytes(json.dumps(review).encode(), "application/json")
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self.send_bytes(str(error).encode(), "text/plain", HTTPStatus.BAD_REQUEST)

    def save_track_label(self) -> None:
        try:
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            clip_id = str(payload["clip_id"])
            track_id = int(payload["track_id"])
            label = str(payload["label"])
            allowed = {
                "team_a",
                "team_b",
                "other",
                "team_a_goalkeeper",
                "team_b_goalkeeper",
            }
            if label not in allowed:
                raise ValueError("Unknown track label")
            if (clip_id, track_id) not in self.server.track_samples:
                raise ValueError("Unknown tracked object")
            key = f"{clip_id}:{track_id}"
            with self.server.label_lock:
                labels = self.server.track_labels()
                labels[key] = {
                    "clip_id": clip_id,
                    "track_id": track_id,
                    "label": label,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self.server.track_labels_path.write_text(
                    json.dumps(
                        {"schema_version": 1, "labels": labels}, indent=2
                    )
                    + "\n"
                )
            self.send_bytes(json.dumps(labels).encode(), "application/json")
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self.send_bytes(str(error).encode(), "text/plain", HTTPStatus.BAD_REQUEST)

    def save_tactic_label(self) -> None:
        try:
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            candidate_id = str(payload["candidate_id"])
            answer = payload["answer"]
            if not isinstance(answer, bool):
                raise ValueError("Answer must be yes or no")
            candidates = {
                row["id"]: row for row in self.server.tactic_candidates()
            }
            if candidate_id not in candidates:
                raise ValueError("Unknown tactical proposal")
            candidate = candidates[candidate_id]
            with self.server.label_lock:
                labels = self.server.tactic_labels()
                labels[candidate_id] = {
                    "candidate_id": candidate_id,
                    "clip_id": candidate["clip_id"],
                    "frame": candidate["frame"],
                    "proposed_label": candidate["label"],
                    "answer": answer,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self.server.tactic_labels_path.write_text(
                    json.dumps(
                        {"schema_version": 1, "labels": labels}, indent=2
                    )
                    + "\n"
                )
            self.send_bytes(json.dumps(labels[candidate_id]).encode(), "application/json")
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self.send_bytes(str(error).encode(), "text/plain", HTTPStatus.BAD_REQUEST)

    def save_synthetic_tactic_label(self) -> None:
        try:
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            sample_id = str(payload["sample_id"])
            answer = payload["answer"]
            if not isinstance(answer, bool):
                raise ValueError("Answer must be yes or no")
            corrected_label = payload.get("corrected_label")
            allowed_corrections = set(self.server.synthetic_label_names) | {"unsure"}
            if answer:
                corrected_label = None
            elif corrected_label not in allowed_corrections:
                raise ValueError("Choose the correct class after answering no")
            prefix, separator, raw_index = sample_id.partition(":")
            if prefix != "synthetic" or not separator:
                raise ValueError("Unknown synthetic graph")
            source_index = int(raw_index)
            if source_index not in self.server.synthetic_indices:
                raise ValueError("Synthetic graph is outside the review set")
            generated_label = self.server.synthetic_label_names[
                int(self.server.synthetic_targets[source_index])
            ]
            with self.server.label_lock:
                labels = self.server.synthetic_tactic_labels()
                labels[sample_id] = {
                    "sample_id": sample_id,
                    "source_index": source_index,
                    "proposed_label": generated_label,
                    "answer": answer,
                    "corrected_label": corrected_label,
                    "training_label": (
                        generated_label if answer else corrected_label
                    ),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self.server.synthetic_tactic_labels_path.write_text(
                    json.dumps(
                        {"schema_version": 1, "labels": labels}, indent=2
                    )
                    + "\n"
                )
            self.send_bytes(json.dumps(labels[sample_id]).encode(), "application/json")
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self.send_bytes(str(error).encode(), "text/plain", HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("artifacts/published-tracking-review"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    server = ReviewServer((args.host, args.port), args.directory)
    print(f"Player tracking review: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
