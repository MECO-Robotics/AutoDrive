"""Small read-only web dashboard for a tensor PPO run."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import re
import time
import secrets
import subprocess


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>REBUILT Gamepiece Training</title><style>
:root{color-scheme:dark;--bg:#10151b;--card:#19222b;--line:#2c3a47;--muted:#a5b2bd;--green:#68e0b0;--blue:#0066B3;--first-red:#ED1C24}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf4f7;font:15px/1.45 system-ui,sans-serif}.wrap{max-width:1160px;margin:auto;padding:28px 20px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}header nav{display:flex;align-items:center;gap:18px}h1{font-size:24px;margin:0}h2{font-size:20px;margin:0 0 12px}h3{font-size:15px;margin:0;color:#c6d8e3}#state{color:var(--green)}.section{margin:0 0 22px}.section-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}.metric{font-size:24px;font-weight:650;margin-top:6px}.label,.sub{color:var(--muted);font-size:13px}.wide{grid-column:span 4}.half{grid-column:span 2}.bar{height:9px;background:#0c1116;border-radius:9px;margin-top:14px;overflow:hidden}.bar i{display:block;width:100%;height:100%;background:var(--green);transform:scaleX(0);transform-origin:left;transition:transform .25s}canvas{display:block;width:100%;background:#111b23;border-radius:8px}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0}.controls input{flex:1 1 150px;min-width:130px}.controls select{max-width:100%;min-width:0}button{background:#293947;color:white;border:1px solid #465a6a;padding:8px 12px;border-radius:7px;cursor:pointer}#chart{height:130px}#field{height:auto;max-height:none;aspect-ratio:auto}.playback-timeline input{flex:1 1 320px;min-width:200px}.active-stack{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.stack-item{padding:6px 10px;border:1px solid #3b5261;border-radius:7px;background:#111b23;font-size:13px}.stack-item strong{color:#dce9ef}.readonly{color:var(--muted);padding:7px 10px;border:1px solid var(--line);border-radius:7px}.error{color:#ffb37e}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:#c6d8e3;font-weight:600}.table-wrap{overflow-x:auto;margin-top:10px}
@media(max-width:700px){.section-grid{grid-template-columns:repeat(2,1fr)}.wide{grid-column:span 2}.half{grid-column:span 2}}
</style></head><body><main class="wrap"><header><h1>REBUILT Gamepiece Training</h1><nav aria-label="Dashboard pages"><a href="/info" style="color:var(--blue)">Field guide</a> <strong id="state">Connecting…</strong></nav></header>
<section class="section" aria-labelledby="liveHeading"><h2 id="liveHeading">Live Simulation</h2><article class="card wide"><div class="active-stack" aria-label="Active controller stack"><span class="stack-item"><strong>ROLE</strong> <span id="activeRole">Offense</span></span><span class="stack-item"><strong>STRATEGY</strong> <span id="activeStrategy">NN Strategy + AD*</span></span><span class="stack-item"><strong>PLANNER</strong> AD*</span><span class="stack-item"><strong>OPPONENT</strong> <span id="activeOpponent">Scripted Defense + AD*</span></span></div><div class="controls"><label for="roleSelector" class="sub">Role</label><select id="roleSelector" aria-label="Robot role"><option value="offense">Offense</option><option value="defense">Defense</option></select><label for="controllerSelector" class="sub">Controller</label><select id="controllerSelector" aria-label="Strategy controller"></select><span class="sub">Motion planner</span><strong class="readonly" aria-label="Motion planner AD star">AD*</strong><label for="opponentSelector" class="sub">Opponent</label><select id="opponentSelector" aria-label="Opponent strategy"></select><label for="playbackSpeed" class="sub">Speed</label><select id="playbackSpeed" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1× · real time</option><option value="2">2×</option><option value="4">4×</option></select><details id="fitnessInspector" hidden><summary>Inspect fitness</summary><select id="fitnessCandidate" aria-label="Candidate fitness by generation and rank"></select><span id="scenarioCount" class="sub"></span></details></div><div class="sub" id="opponentHelp">In this simulator, deterministic offense uses scripted objective selection with AD* motion.</div><div class="sub" id="bestPlayback">Choose a role, strategy, and opponent.</div><div class="sub" id="opponentFitness">Select a recorded matchup to view its rollout.</div><div class="controls playback-timeline"><label for="frame" class="sub">Playback position</label><input id="frame" type="range" min="0" max="0" value="0" aria-label="Playback position"><span id="frameCount" class="sub">0 / 0</span></div><select id="attackerMode" hidden aria-hidden="true"><option value="nn">NN</option><option value="adstar">AD*</option><option value="mixed">Mixed</option></select><select id="defenderMode" hidden aria-hidden="true"><option value="nn">NN</option><option value="guard">Scripted</option><option value="adstar_defender">Reactive AD*</option><option value="mixed">Mixed</option></select><select id="zoneScenario" hidden aria-label="Start and goal zones"><option value="">Recorded scenarios</option><option value="red-center">Red → Center</option><option value="red-blue">Red → Blue</option><option value="center-red">Center → Red</option><option value="center-blue">Center → Blue</option><option value="blue-red">Blue → Red</option><option value="blue-center">Blue → Center</option></select><select id="generation" aria-label="Playback generation" hidden></select><select id="scenario" aria-label="Playback scenario" hidden><option value="">Scenario</option></select><button id="play" hidden>Pause</button><input id="showGhosts" type="checkbox" checked hidden><input id="showGrid" type="checkbox" checked hidden><input id="showFuel" type="checkbox" checked hidden><div id="matchupNotice" class="sub" role="status" aria-live="polite"></div><div id="gameState" class="sub" aria-live="polite">FUEL: awaiting playback frame</div><canvas id="field" width="1100" height="536"></canvas><p class="sub"><strong style="color:#0066B3">Blue:</strong> defender · <strong style="color:#ED1C24">Red:</strong> attacker · Gold front bar: intake side · Dotted warm/cool lines: available AD* routes · White arrows: heading / robot effort · Gold dots: loose FUEL · Warm/cool dots: held by robots · Hub rings: active or inactive.</p></article></section>
<section class="section" aria-labelledby="comparisonHeading"><h2 id="comparisonHeading">Controller Comparison</h2><article class="card"><p class="sub">Strategy pairs are evaluated on identical seeded scenarios with the same episode count and match horizon. Compare results directly when Scenario comparability is marked “matched”; unpaired results are labeled and should not be compared directly.</p><div class="table-wrap"><table id="ablations"><thead><tr><th>Strategy</th><th>Role</th><th>Episodes</th><th>Acquisitions</th><th>Scores</th><th>Passes (attempted / completed)</th><th>Cycle time (s)</th><th>Defensive delay (s)</th><th>Denied / abandoned</th><th>Contacts</th><th>Total FUEL scored</th><th>Scenario comparability</th><th>Source</th></tr></thead><tbody><tr><td colspan="13" class="sub">Loading comparison…</td></tr></tbody></table></div></article></section>
<section class="section" aria-labelledby="trainingHeading"><h2 id="trainingHeading">Training</h2><div class="section-grid"><article class="card"><div class="label" id="progressLabel">Progress</div><div class="metric" id="progress">—</div><div class="bar"><i id="fill"></i></div></article><article class="card"><div class="label">Throughput</div><div class="metric" id="rate">—</div><div class="sub">transitions / second</div></article><article class="card"><div class="label">Accelerator</div><div class="metric" id="device">—</div><div class="sub" id="backend"></div></article><article class="card"><div class="label">Training setup</div><div class="metric" id="setup">—</div><div class="sub" id="opp"></div><div class="sub" id="drivetrain">Drivetrain: —</div><div class="sub" id="curriculum"></div></article><article class="card wide"><div class="label">Training timeline</div><canvas id="chart" width="1100" height="145"></canvas></article><article class="card half"><div class="label">Training epochs</div><div id="trainingEpochs" class="metric">—</div><div id="trainingRun" class="sub"></div></article><article class="card half"><div class="label">Training time</div><div id="updated" class="sub">—</div></article></div></section>
<section class="section" aria-labelledby="evaluationHeading"><h2 id="evaluationHeading">Evaluation</h2><article class="card"><div class="controls"><label for="gameEvaluation" class="sub">Game evaluation</label><select id="gameEvaluation" aria-label="Game evaluation playback"><option value="">Select evaluation</option></select></div><div id="evaluation" class="sub">Choose an evaluation to load its match playback.</div><div class="sub">Playback opens in Live Simulation above and includes gamepiece locations and scoring state.</div></article></section></main>
<script>
const $=id=>document.getElementById(id), chart=$('chart'), cx=chart.getContext('2d'), field=$('field'), fx=field.getContext('2d');
const initialRun=new URLSearchParams(location.search).get('run');
let history=[],frames=[],index=0,playing=true,playbackSpeed=1,playbackTimes=[],clockFrames=null,playbackAnchorWall=performance.now(),playbackAnchorSim=0,playbackSimTime=0,frameAlpha=0,world={length:16.54,width:8.07,colliders:[]},playbackRevision=null,playbackTask='',loadedScenarios=[],generationCatalog=[],generationCache={},playbackRuns=[],selectedRun='',selectionRevision=0,pollTimer=null,zoneLoading=false,playbackTransition=false,autoBestKey='';
function ensurePlaybackClock(){if(clockFrames===frames)return;clockFrames=frames;playbackTimes=frames.map((frame,i)=>Number.isFinite(Number(frame.match_elapsed))?Number(frame.match_elapsed):frames.length>1?160*i/(frames.length-1):0);for(let i=1;i<playbackTimes.length;i++)playbackTimes[i]=Math.max(playbackTimes[i],playbackTimes[i-1]);playbackAnchorSim=playbackTimes[0]||0;playbackAnchorWall=performance.now();playbackSimTime=playbackAnchorSim;frameAlpha=0}
function setPlaybackTime(seconds){ensurePlaybackClock();if(!playbackTimes.length)return;playbackSimTime=Math.max(playbackTimes[0],Math.min(playbackTimes[playbackTimes.length-1],seconds));let lo=0,hi=playbackTimes.length-1;while(lo<hi){const mid=(lo+hi+1)>>1;if(playbackTimes[mid]<=playbackSimTime)lo=mid;else hi=mid-1}index=lo;const next=Math.min(index+1,playbackTimes.length-1),span=playbackTimes[next]-playbackTimes[index];frameAlpha=span>0?(playbackSimTime-playbackTimes[index])/span:0}
function restartPlaybackClock(){ensurePlaybackClock();index=0;frameAlpha=0;playbackSimTime=playbackTimes[0]||0;playbackAnchorSim=playbackSimTime;playbackAnchorWall=performance.now()}
function currentPlaybackFrame(){const f=frames[index];if(!f)return f;const next=frames[Math.min(index+1,frames.length-1)];if(!next||frameAlpha<=0||!f.robots||!next.robots)return f;const robots=f.robots.map((r,k)=>{const n=next.robots[k];if(!r||!n)return r;const d=Math.atan2(Math.sin(n[2]-r[2]),Math.cos(n[2]-r[2]));return [r[0]+(n[0]-r[0])*frameAlpha,r[1]+(n[1]-r[1])*frameAlpha,r[2]+d*frameAlpha]});return {...f,robots,match_elapsed:playbackSimTime}}
function zoneName(x){const az=world.alliance_zone_depth||4.028;return x<=az?'Red':x>=world.length-az?'Blue':'Center'}
function roleLabel(task){return task==='counter_defense'?'Offense':task==='defense'?'Defense':'REBUILT gamepiece training'}function strategyName(value){return ({guard:'Scripted Defense + AD*',adstar_defender:'Reactive AD* Defense',adstar:'AD* Offense',learned:'NN Strategy + AD*',nn:'NN Strategy + AD*',scripted_offense_adstar:'Scripted Strategy + AD*',learned_offense_adstar:'NN Strategy + AD*',scripted_defense_adstar:'Scripted Defense + AD*',learned_defense_adstar:'NN Strategy + AD*'})[String(value||'').toLowerCase()]||String(value||'—').replaceAll('_',' ')}function displayMatchup(value){return String(value||'').replace(/adstar_defender/gi,'Reactive AD* Defense').replace(/adstar/gi,'AD*').replace(/guard/gi,'Scripted Defense + AD*').replace(/learned/gi,'NN Strategy').replace(/counter_defense/gi,'Offense').replace(/defense/gi,'Defense').replace(/_/g,' ')}function trainingTaskLabel(status){return roleLabel(status?.task)}
function updateScenarioCount(){const i=loadedScenarios.findIndex(s=>s.id===$('scenario').value);const s=loadedScenarios[i];const byOpponent=s?.opponent_fitness?Object.entries(s.opponent_fitness).map(([k,v])=>`${k} ${Number(v).toFixed(2)}`).join(' · '):'';$('scenarioCount').textContent=s?(s.rank?`Rank ${s.rank} of ${loadedScenarios.length} · fitness ${Number(s.fitness).toFixed(3)}${byOpponent?' · '+byOpponent:''}`:`${s.label||`Scenario ${i+1}`} · ${i+1} of ${loadedScenarios.length}${s.routeLabel?' · '+s.routeLabel:''}`):''}
function updateFitnessOptions(){const options=generationCatalog.flatMap(g=>(g.candidates||[]).map(c=>({value:`${g.generation}:${c.rank}`,label:`Gen ${g.generation} · rank ${c.rank} · fitness ${Number(c.fitness).toFixed(3)}`})));const select=$('fitnessCandidate'),previous=select.value;select.replaceChildren(...options.map(item=>new Option(item.label,item.value)));select.value=options.some(item=>item.value===previous)?previous:options[0]?.value||'';$('fitnessInspector').hidden=!options.length}
async function displayFitnessCandidate(generation,rank,run=selectedRun,revision=selectionRevision){await loadGeneration(generation,false,run,revision);if(revision!==selectionRevision||run!==selectedRun)return;const scenario=loadedScenarios.find(item=>item.rank===rank);if(!scenario)return;$('scenario').value=scenario.id;frames=scenario.frames||[];index=0;updateScenarioCount();$('bestPlayback').textContent=`Candidate playback · generation ${generation} · rank ${rank} · fitness ${Number(scenario.fitness).toFixed(3)}`;drawField()}
async function selectFitnessCandidate(){const [generation,rank]=($('fitnessCandidate').value||'').split(':').map(Number);if(!generation||!rank)return;try{await displayFitnessCandidate(generation,rank)}catch(error){$('matchupNotice').textContent=`Could not load candidate fitness: ${error.message}`}}
function renderAblations(records){const body=$('ablations').querySelector('tbody'),expected=[['scripted_offense_adstar','Scripted Strategy + AD*'],['learned_offense_adstar','NN Strategy + AD*'],['scripted_defense_adstar','Scripted Defense + AD*'],['learned_defense_adstar','NN Strategy + AD*']],key=record=>String(record.name||record.architecture||record.ablation||'').toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,''),matched=new Set(),rows=expected.map(([id,label])=>{const record=records.find(item=>key(item)===id);if(record)matched.add(record);return{...record,name:label,canonical_name:id,no_data:!record||record.evaluated===false}});const extras=records.filter(record=>!matched.has(record));const fmt=value=>value==null||value===''?'—':typeof value==='number'?value.toFixed(2):String(value),gameValue=(record,keys)=>{for(const key of keys)if(record[key]!=null)return record[key];return null};body.replaceChildren(...[...rows,...extras].map(record=>{const row=document.createElement('tr'),denied=gameValue(record,['mean_denied_objectives','mean_denied_objective_count','denied_objectives','denied']),abandoned=gameValue(record,['mean_abandoned_objectives','mean_abandoned_objective_count','abandoned_objectives','abandoned']),comparability=record.no_data?(record.missing_reason?'Not evaluable':'No evaluation'):record.scenario_comparability??(record.scenario_comparable===true||record.comparable===true?'Matched':record.scenario_comparable===false||record.comparable===false?'Not paired':record.seeds?.length?`Seed set · ${record.seeds.length}`:'—'),role=roleLabel(record.task||((record.canonical_name||'').includes('offense')?'counter_defense':'defense')),values=[record.name||strategyName(record.architecture)||'Strategy',role,record.no_data?'—':record.episodes??'—',record.no_data?'—':fmt(gameValue(record,['mean_acquisitions','mean_acquisitions_per_episode','acquisitions_per_episode','acquisitions','total_acquisitions'])),record.no_data?'—':fmt(gameValue(record,['mean_scores','mean_score_count','scores','scored_pieces','total_scored'])),record.no_data?'—':`${fmt(gameValue(record,['mean_passes_attempted']))} / ${fmt(gameValue(record,['mean_passes_completed']))}`,record.no_data?'—':fmt(gameValue(record,['mean_cycle_time','mean_cycle_time_seconds','cycle_time','mean_time_to_goal'])),record.no_data?'—':fmt(gameValue(record,['mean_defensive_delay','mean_blocking_delay_s','defensive_delay','mean_blocking_delay'])),record.no_data?'—':`${fmt(denied)} / ${fmt(abandoned)}`,record.no_data?'—':fmt(gameValue(record,['mean_contacts','contacts','mean_contact_count','mean_contact'])),record.no_data?'—':fmt(gameValue(record,['mean_total_simulated_score','mean_total_score','total_score','total_scoring','total_points','mean_score'])),comparability,record.no_data?(record.missing_reason||'—'):`${record.run_name||'metrics'}${record.source?` · ${record.source}`:''}`];for(const value of values){const cell=document.createElement('td');cell.textContent=String(value);row.append(cell)}return row}))}
async function loadAblations(){try{const response=await fetch('/api/ablations',{cache:'no-store'});if(response.ok)renderAblations(await response.json())}catch(_error){}}
async function loadGameEvaluations(){try{const response=await fetch('/api/game-evaluations',{cache:'no-store'});if(!response.ok)return;const entries=await response.json(),select=$('gameEvaluation'),previous=select.value;select.replaceChildren(new Option('Select evaluation',''),...entries.map(entry=>new Option(entry.label,entry.id)));select.value=entries.some(entry=>entry.id===previous)?previous:''}catch(_error){}}
async function selectGameEvaluation(){const name=$('gameEvaluation').value;if(!name)return;selectionRevision++;selectedRun='';clearTimeout(pollTimer);resetPlayback();$('zoneScenario').value='';$('matchupNotice').textContent='Loading game-piece evaluation playback…';try{const response=await fetch(`/api/game-evaluation-playback?name=${encodeURIComponent(name)}`,{cache:'no-store'}),record=await response.json();if(!response.ok)throw new Error(record.error||`Playback request failed (${response.status})`);playbackTask=record.task||'';world=record.field||world;loadedScenarios=record.scenarios?.length?record.scenarios:[{id:'evaluation',label:'Evaluation',frames:record.frames||[]}];if(record.frames?.length&&loadedScenarios[0].frames?.length===record.frames.length)loadedScenarios[0]={...loadedScenarios[0],frames:loadedScenarios[0].frames.map((frame,i)=>({...frame,fuel_pieces:record.frames[i].fuel_pieces||[]}))};const scenario=loadedScenarios[0];frames=scenario.frames||[];index=0;$('scenario').replaceChildren(...loadedScenarios.map(s=>new Option(s.label||s.id,s.id)));$('scenario').dataset.catalog=`game-evaluation:${name}`;$('scenario').value=scenario.id;$('scenarioCount').textContent=`${loadedScenarios.length} rollout${loadedScenarios.length===1?'':'s'}`;$('showGhosts').checked=false;$('frame').max=Math.max(0,frames.length-1);$('matchupNotice').textContent=$('gameEvaluation').selectedOptions[0].textContent+' · FUEL field-state playback';$('bestPlayback').textContent='2026 game evaluation';drawField()}catch(error){$('matchupNotice').textContent=`Could not load game evaluation: ${error.message}`}}
function drawChart(){const w=chart.width,h=chart.height;cx.clearRect(0,0,w,h);cx.strokeStyle='#31414e';cx.beginPath();cx.moveTo(32,10);cx.lineTo(32,h-22);cx.lineTo(w-8,h-22);cx.stroke();if(history.length<2)return;let max=Math.max(...history,1);cx.strokeStyle='#68e0b0';cx.lineWidth=2;cx.beginPath();history.forEach((v,i)=>{let x=32+i*(w-44)/(history.length-1),y=h-22-v/max*(h-38);i?cx.lineTo(x,y):cx.moveTo(x,y)});cx.stroke();}
function drawAdstarRoutes(frame,attackerIndex,opacity,ox,s,py){if(!frame?.robots)return;const paths=Array.isArray(frame.adstar_paths)&&frame.adstar_paths.length===2?frame.adstar_paths:[[],[]];for(let k=0;k<2;k++){const route=paths[k];if(!Array.isArray(route)||route.length<2)continue;fx.save();fx.globalAlpha=opacity;fx.beginPath();route.forEach((point,i)=>{const x=ox+point[0]*s,y=py(point[1]);i?fx.lineTo(x,y):fx.moveTo(x,y)});fx.setLineDash([2,7]);fx.lineCap='round';fx.strokeStyle=k===attackerIndex?'#ff9d91':'#80caff';fx.lineWidth=3;fx.stroke();fx.restore()}}
function drawField(){
 const w=field.width,h=field.height,pad=32,s=Math.min((w-2*pad)/world.length,(h-2*pad)/world.width),ox=(w-world.length*s)/2,oy=(h-world.width*s)/2;
 const py=y=>oy+(world.width-y)*s,az=world.alliance_zone_depth||4.028;
 fx.clearRect(0,0,w,h);fx.fillStyle='#252e32';fx.fillRect(ox,oy,world.length*s,world.width*s);
 fx.fillStyle='rgba(198,93,77,.18)';fx.fillRect(ox,oy,az*s,world.width*s);
 fx.fillStyle='rgba(76,126,180,.18)';fx.fillRect(ox+(world.length-az)*s,oy,az*s,world.width*s);
 fx.strokeStyle='#e1e6e3';fx.lineWidth=2;fx.strokeRect(ox,oy,world.length*s,world.width*s);
 // FIRST field markings: alliance-zone boundaries and field centerline.
 fx.setLineDash([7,6]);fx.lineWidth=1.5;fx.strokeStyle='#d8dedc';
 for(const x of [az,world.length/2,world.length-az]){fx.beginPath();fx.moveTo(ox+x*s,oy);fx.lineTo(ox+x*s,oy+world.width*s);fx.stroke()}fx.setLineDash([]);
 if($('showGrid').checked){
  const cell=.25;fx.save();fx.beginPath();fx.rect(ox,oy,world.length*s,world.width*s);fx.clip();
  fx.strokeStyle='rgba(177,202,216,.20)';fx.lineWidth=1;
  for(let x=0;x<=world.length+1e-6;x+=cell){const px=ox+x*s;fx.beginPath();fx.moveTo(px,oy);fx.lineTo(px,oy+world.width*s);fx.stroke()}
  for(let y=0;y<=world.width+1e-6;y+=cell){const pyCell=oy+(world.width-y)*s;fx.beginPath();fx.moveTo(ox,pyCell);fx.lineTo(ox+world.length*s,pyCell);fx.stroke()}
  fx.restore();
 }
 fx.fillStyle='#ED1C24';fx.font='bold 12px system-ui';fx.fillText('RED ALLIANCE',ox+10,oy+17);
 fx.fillStyle='#0066B3';fx.fillText('BLUE ALLIANCE',ox+world.length*s-111,oy+17);
 for(const b of world.elements||world.colliders||[]){
  const x=ox+(b.x-b.length/2)*s,y=py(b.y+b.width/2),bw=b.length*s,bh=b.width*s;
  if(b.name?.includes('_trench_')&&!b.name.includes('_support_')){
   fx.fillStyle='rgba(113,135,122,.10)';fx.fillRect(x,y,bw,bh);fx.save();fx.setLineDash([5,5]);fx.strokeStyle='#a9c3b2';fx.lineWidth=1.5;fx.strokeRect(x,y,bw,bh);fx.restore();
   fx.strokeStyle='#a9c3b2';fx.lineWidth=3;fx.beginPath();fx.moveTo(x+bw/2,y);fx.lineTo(x+bw/2,y+bh);fx.stroke();continue;
  }
  fx.fillStyle=b.color?.includes('support')?'#b09a72':b.color?.includes('hub')?'#707a7e':b.color?.includes('bump')?(b.color.includes('red')?'#ED1C24':'#0066B3'):b.color?.includes('tower')?(b.color.includes('red')?'#ED1C24':'#0066B3'):b.color?.includes('depot')?'#c1a96f':'#59675d';
  fx.fillRect(x,y,bw,bh);fx.strokeStyle='#e1e6e3';fx.lineWidth=b.color?.includes('support')?1.7:1.2;fx.strokeRect(x,y,bw,bh);
  if(b.name?.includes('hub')){
   const cx=x+bw/2,cy=y+bh/2,r=Math.min(bw,bh)*.443;fx.beginPath();
   for(let i=0;i<6;i++){const a=-Math.PI/2+i*Math.PI/3,px=cx+r*Math.cos(a),pY=cy+r*Math.sin(a);i?fx.lineTo(px,pY):fx.moveTo(px,pY)}
   fx.closePath();fx.fillStyle='#273238';fx.fill();fx.strokeStyle='#f1d98d';fx.lineWidth=2;fx.stroke();
   fx.fillStyle='#f3e7bc';fx.font='bold 10px system-ui';fx.textAlign='center';fx.textBaseline='middle';fx.fillText('72″',cx,cy);fx.textAlign='start';fx.textBaseline='alphabetic';
  }
 }
 if(!frames.length)return;
 const f=currentPlaybackFrame(),defensePlayback=playbackTask==='defense'||playbackTask==='adstar_attacker_defense',attackerIndex=defensePlayback?1:0,defenderIndex=1-attackerIndex;drawAdstarRoutes(f,attackerIndex,1,ox,s,py);

 if(Array.isArray(f.hub_centers)){for(let i=0;i<f.hub_centers.length;i++){const hub=f.hub_centers[i],active=f.hub_active?.[i]!==false;fx.beginPath();fx.arc(ox+hub[0]*s,py(hub[1]),Math.max(4,.58*s),0,Math.PI*2);fx.strokeStyle=active?'#f3e7bc':'#77818a';fx.lineWidth=2;fx.stroke()}}
 const pieces=Array.isArray(f.fuel_pieces)?f.fuel_pieces:[];let loose=0,held0=0,held1=0;
 for(const piece of pieces){if(!piece||piece.length<3)continue;const owner=piece[2],held=owner>=0;if(owner===-1)loose++;else if(owner===0)held0++;else if(owner===1)held1++;if(!$('showFuel').checked)continue;fx.beginPath();fx.arc(ox+piece[0]*s,py(piece[1]),held?5.2:4.4,0,Math.PI*2);fx.fillStyle=owner===0?'#ff9d91':owner===1?'#80caff':'#ffd84d';fx.fill();fx.strokeStyle='#18222a';fx.lineWidth=1.5;fx.stroke()}
 $('gameState').textContent=`FUEL in frame: ${pieces.length} active · loose ${loose} · held by robot 0 ${held0} · held by robot 1 ${held1}${f.match_remaining!=null?` · ${Number(f.match_remaining).toFixed(1)} s remaining`:''}${f.fuel_score_count?.length?` · scored ${f.fuel_score_count.join(' / ')}`:''}`;
 if(f.predicted_intercept){const x=ox+f.predicted_intercept[0]*s,y=py(f.predicted_intercept[1]);fx.save();fx.translate(x,y);fx.rotate(Math.PI/4);fx.fillStyle='#ffe071';fx.strokeStyle='#302a16';fx.lineWidth=1.5;fx.fillRect(-7,-7,14,14);fx.strokeRect(-7,-7,14,14);fx.restore();fx.fillStyle='#ffe071';fx.font='12px system-ui';fx.fillText(`intercept ${Number(f.predicted_intercept_time||0).toFixed(1)}s`,x+10,y-9)}
 const trailFrames=frames.slice(Math.max(0,index-109),index+1);
 const official={attacker:'#ED1C24',defender:'#0066B3'};
 if($('showGhosts').checked){const selectedId=$('scenario').value,progress=frames.length>1?index/(frames.length-1):0;for(const scenario of loadedScenarios){if(scenario.id===selectedId||!scenario.frames?.length)continue;const ghostFrames=scenario.frames,ghostIndex=Math.min(ghostFrames.length-1,Math.round(progress*Math.max(0,ghostFrames.length-1))),ghost=ghostFrames[ghostIndex];if(!ghost?.robots)continue;
  // Each ghost keeps its own motion history and planner route, synchronized
  // to the selected run by normalized playback progress.
  const ghostTrail=ghostFrames.slice(Math.max(0,ghostIndex-109),ghostIndex+1);
  for(let k=0;k<ghost.robots.length;k++){const color=k===attackerIndex?'#ED1C24':'#0066B3';fx.save();fx.globalAlpha=.45;fx.strokeStyle=color;fx.lineWidth=2;fx.beginPath();let started=false;for(const frame of ghostTrail){const point=frame.robots?.[k];if(!point)continue;const x=ox+point[0]*s,y=py(point[1]);if(!started){fx.moveTo(x,y);started=true}else fx.lineTo(x,y)}fx.stroke();fx.restore()}
  drawAdstarRoutes(ghost,attackerIndex,.68,ox,s,py)
  for(let k=0;k<ghost.robots.length;k++){const r=ghost.robots[k],[l,b]=ghost.sizes?.length>=k*2+2?ghost.sizes.slice(k*2,k*2+2):f.sizes.slice(k*2,k*2+2),color=k===attackerIndex?'#ED1C24':'#0066B3';fx.save();fx.globalAlpha=.52;fx.translate(ox+r[0]*s,py(r[1]));fx.rotate(-r[2]);fx.fillStyle=color+'35';fx.strokeStyle=color;fx.lineWidth=2;fx.setLineDash([5,4]);fx.fillRect(-l*s/2,-b*s/2,l*s,b*s);fx.strokeRect(-l*s/2,-b*s/2,l*s,b*s);fx.setLineDash([]);fx.beginPath();fx.moveTo(0,0);fx.lineTo(l*s*.48,0);fx.stroke();fx.restore()}
 }}
 for(let k=0;k<2;k++){const color=k===attackerIndex?official.attacker:official.defender;let segment=[];const drawSegment=()=>{if(segment.length>1){fx.beginPath();segment.forEach((p,i)=>{const x=ox+p[0]*s,y=py(p[1]);i?fx.lineTo(x,y):fx.moveTo(x,y)});fx.stroke()}};fx.strokeStyle=color+'88';fx.lineWidth=2;for(const frame of trailFrames){const point=frame.robots?.[k]?.slice(0,2);if(!point||point.length<2)continue;const previous=segment[segment.length-1];if(previous&&Math.hypot(point[0]-previous[0],point[1]-previous[1])>.8){drawSegment();segment=[]}segment.push(point)}drawSegment()}
 f.robots.forEach((r,k)=>{let [x,y,t]=r,[l,b]=f.sizes.slice(k*2,k*2+2);fx.save();fx.translate(ox+x*s,py(y));fx.rotate(-t);fx.fillStyle=k===attackerIndex?official.attacker:official.defender;fx.fillRect(-l*s/2,-b*s/2,l*s,b*s);fx.lineJoin='round';fx.strokeStyle='#10151b';fx.lineWidth=7;fx.strokeRect(-l*s/2,-b*s/2,l*s,b*s);fx.strokeStyle='#f4f7f6';fx.lineWidth=2.5;fx.strokeRect(-l*s/2,-b*s/2,l*s,b*s);fx.beginPath();fx.moveTo(l*s*.43,-b*s*.32);fx.lineTo(l*s*.43,b*s*.32);fx.strokeStyle='#161b1e';fx.lineWidth=7;fx.stroke();fx.strokeStyle='#ffd84d';fx.lineWidth=4;fx.stroke();fx.beginPath();fx.moveTo(0,0);fx.lineTo(l*s*.52,0);fx.strokeStyle='#f4f7f6';fx.lineWidth=2.5;fx.stroke();fx.restore()});
 f.robots.forEach((r,k)=>{let vectors=f.robot_effort_vectors;let v=Array.isArray(vectors?.[k])?vectors[k]:(k===defenderIndex?f.chassis_effort_vector:null);if(!Array.isArray(v)||v.length<2||!v.every(Number.isFinite)){const before=frames[Math.max(0,index-1)]?.robots?.[k],after=frames[Math.min(frames.length-1,index+1)]?.robots?.[k];v=before&&after?[after[0]-before[0],after[1]-before[1]]:[0,0];v=v.map(x=>x*3)}const mag=Math.hypot(v[0],v[1]);if(mag>.03){const len=Math.min(1.15,mag*.28)*s,dx=v[0]/mag*len,dy=-v[1]/mag*len,x=ox+r[0]*s,y=py(r[1]),ang=Math.atan2(dy,dx);fx.save();fx.strokeStyle='#FFFFFF';fx.fillStyle='#FFFFFF';fx.lineWidth=4;fx.lineCap='round';fx.beginPath();fx.moveTo(x,y);fx.lineTo(x+dx,y+dy);fx.stroke();fx.translate(x+dx,y+dy);fx.rotate(ang);fx.beginPath();fx.moveTo(0,0);fx.lineTo(-11,-6);fx.lineTo(-11,6);fx.closePath();fx.fill();fx.restore()}})
 $('frame').max=Math.max(0,frames.length-1);$('frame').value=index;$('frameCount').textContent=frames.length?`${index+1} / ${frames.length} · ${playbackSimTime.toFixed(1)} / ${playbackTimes[playbackTimes.length-1]?.toFixed(1)||'0.0'} s`:'Waiting for rollout frames';
}

function elapsedLabel(seconds){if(!Number.isFinite(seconds)||seconds<0)return '—';const total=Math.floor(seconds),hours=Math.floor(total/3600),minutes=Math.floor((total%3600)/60),secs=total%60;return hours?`${hours}h ${minutes}m ${secs}s`:`${minutes}m ${secs}s`}
async function loadGeneration(generation,selectBest=true,run=selectedRun,revision=selectionRevision){let record=generationCache[generation];if(!record){const response=await fetch(`/api/generation?run=${run}&generation=${generation}`,{cache:'no-store'});if(!response.ok)throw new Error(`Generation request failed (${response.status})`);record=await response.json();if(revision!==selectionRevision||run!==selectedRun)return;generationCache[generation]=record}if(revision!==selectionRevision||run!==selectedRun)return;playbackTask=record.task||playbackTask;world=record.field||world;loadedScenarios=(record.candidates||[]).map(c=>({id:c.id||`rank-${c.rank}`,label:c.label,rank:c.rank,fitness:c.fitness,opponent_fitness:c.opponent_fitness,opponent_metrics:c.opponent_metrics,frames:c.frames||[]}));const catalog=loadedScenarios.map(s=>`${s.id}:${s.fitness}`).join('|');if($('scenario').dataset.catalog!==catalog){$('scenario').replaceChildren(...loadedScenarios.map(s=>{const o=document.createElement('option');o.value=s.id;o.textContent=s.label||`Rank ${s.rank} · fitness ${Number(s.fitness).toFixed(3)}`;return o}));$('scenario').dataset.catalog=catalog}const keep=loadedScenarios.some(s=>s.id===$('scenario').value);if(selectBest||!keep)$('scenario').value=loadedScenarios[0]?.id||'';frames=(loadedScenarios.find(s=>s.id===$('scenario').value)||loadedScenarios[0])?.frames||[];index=0;$('showGhosts').checked=false;updateScenarioCount();drawField()}
async function refreshPlayback(run,selectionId=selectionRevision){const headers=playbackRevision?{'If-None-Match':playbackRevision}:{};const response=await fetch(`/api/playback?run=${run}`,{cache:'no-store',headers});if(selectionId!==selectionRevision||run!==selectedRun)return;if(response.status===304)return;const revision=response.headers.get('ETag')||response.headers.get('X-Playback-Revision'),p=await response.json(),changed=!!revision&&revision!==playbackRevision;if(revision)playbackRevision=revision;if(p.generations?.length){generationCatalog=p.generations;updateFitnessOptions();$('fitnessInspector').hidden=false;$('generation').hidden=false;const catalog=generationCatalog.map(x=>x.generation).join(',');if($('generation').dataset.catalog!==catalog){$('generation').replaceChildren(...generationCatalog.map(x=>{const o=document.createElement('option');o.value=x.generation;o.textContent=`Generation ${x.generation}`;return o}));$('generation').dataset.catalog=catalog}const latest=Number(generationCatalog[generationCatalog.length-1].generation),bestGeneration=Number(p.best_population?.generation)||latest,bestFitness=Number(p.best_population?.fitness),bestSummary=generationCatalog.find(item=>Number(item.generation)===bestGeneration),bestCandidate=bestSummary?.candidates?.find(item=>Number.isFinite(bestFitness)&&Math.abs(Number(item.fitness)-bestFitness)<1e-8),bestRank=Number(bestCandidate?.rank)||1,bestKey=`${bestGeneration}:${bestRank}:${Number.isFinite(bestFitness)?bestFitness:'unknown'}`;$('generation').value=String(bestGeneration);if(bestKey!==autoBestKey||!frames.length){$('fitnessCandidate').value=`${bestGeneration}:${bestRank}`;await displayFitnessCandidate(bestGeneration,bestRank,run,selectionId);autoBestKey=bestKey}$('bestPlayback').textContent=`Global best · generation ${bestGeneration} · fitness ${Number.isFinite(bestFitness)?bestFitness.toFixed(3):'—'}${p.best_population?.run_score==null?'':` · rollout ${Number(p.best_population.run_score).toFixed(2)}`}`;return}generationCatalog=[];$('generation').hidden=true;$('fitnessInspector').hidden=true;playbackTask=p.task||'';world=p.field||world;loadedScenarios=p.scenarios?.length?p.scenarios:[{id:'default',label:'Scenario 1',frames:p.frames||[]}];for(const [i,s] of loadedScenarios.entries()){const f=s.frames?.[0],start=s.start?.[0]??(p.task==='counter_defense'?f?.robots?.[0]?.[0]:f?.robots?.[1]?.[0]),goal=(s.goal||f?.goal)?.[0];s.routeLabel=Number.isFinite(start)&&Number.isFinite(goal)?`${zoneName(start)} → ${zoneName(goal)}`:'';s.displayLabel=`${s.label||`Scenario ${i+1}`}${s.routeLabel?' · '+s.routeLabel:''}`}const scenarioCatalog=loadedScenarios.map(s=>`${s.id}:${s.displayLabel}`).join('|');if($('scenario').dataset.catalog!==scenarioCatalog){const previous=$('scenario').value;$('scenario').replaceChildren(...loadedScenarios.map(s=>{const o=document.createElement('option');o.value=s.id;o.textContent=s.displayLabel;return o}));$('scenario').dataset.catalog=scenarioCatalog;$('scenario').value=loadedScenarios.some(s=>s.id===previous)?previous:loadedScenarios[0]?.id||'';if(!changed&&$('scenario').value!==previous)index=0}const selected=loadedScenarios.find(s=>s.id===$('scenario').value)||loadedScenarios[0];frames=selected?.frames||[];if(changed)index=0;if(frames.length&&index>=frames.length)index=frames.length-1;updateScenarioCount();drawField();if(p.best_population&&run!=='adstar-attacker-defense')$('bestPlayback').textContent=`Best population generation ${p.best_population.generation} · fitness ${Number(p.best_population.fitness).toFixed(3)} · best run score ${Number(p.best_population.run_score).toFixed(2)}`}
function schedulePoll(){clearTimeout(pollTimer);pollTimer=setTimeout(poll,1500)}async function poll(){const run=selectedRun,revision=selectionRevision;if(!run){$('state').textContent='No playback available';schedulePoll();return}try{let [r,tr]=await Promise.all([fetch('/api/status?run='+run,{cache:'no-store'}),fetch('/api/training-status',{cache:'no-store'})]),d=await r.json(),training=await tr.json();if(revision!==selectionRevision||run!==selectedRun)return;if(training.algorithm==='generational'&&training.status!=='waiting')d=training;$('state').textContent=d.status==='running'&&d.architecture==='strategic_adstar'?`Training ${trainingTaskLabel(d)} · active`:d.status==='completed'?'Training complete':d.status||'Waiting';$('progressLabel').textContent=d.algorithm==='generational'?`Generation ${d.current_generation||d.generation||0} / ${d.total_generations||0}`:d.algorithm==='crossplay'?'Playback frames':d.progress_unit==='episodes'?'Evaluation episodes':'Progress';$('progress').textContent=`${(d.completed_timesteps||0).toLocaleString()} / ${(d.requested_timesteps||1).toLocaleString()}`;$('fill').style.transform=`scaleX(${Math.min(1,(d.completed_timesteps||0)/(d.requested_timesteps||1))})`;$('rate').textContent=Math.round(d.transitions_per_second||0).toLocaleString();$('device').textContent=d.device_name||d.device||'—';$('backend').textContent=d.accelerator_backend||'';$('setup').textContent=d.algorithm==='crossplay'?`${d.matchup||'NN vs NN'} · ${d.num_envs||0} scenarios`:`${trainingTaskLabel(d)} · ${d.num_envs||0} envs${d.algorithm==='generational'?` · population ${d.population_size||0}`:''}`;$('opp').textContent=d.algorithm==='crossplay'?`Opponent policies: ${strategyName(d.opponent||d.matchup||'NN')}`:`Opponent: ${strategyName(d.opponent||'—')}${d.opponents?.length>1?' · '+d.opponents.map(strategyName).join(', '):''}`;const oppFitness=d.opponent_fitness||{},oppMetrics=d.opponent_metrics||{};$('opponentFitness').textContent=Object.keys(oppFitness).length?'Current generation best: '+Object.entries(oppFitness).map(([k,v])=>`${strategyName(k)} fit ${Number(v).toFixed(2)}${oppMetrics[k]?` · hold ${Math.round(100*oppMetrics[k].hold_rate)}%`:''}`).join(' · '):d.algorithm==='crossplay'?`Defender and attacker policy checkpoints are fixed across matchups.`:'Opponent-specific fitness: waiting for training';const dc=d.drivetrain_config;$('drivetrain').textContent=dc?`Drivetrain: ${dc.name||'configured robot'}${dc.randomize?' · randomized':' · fixed parameters'}`:'Drivetrain config unavailable';$('curriculum').textContent=d.algorithm==='crossplay'?`Opponent matchup: ${displayMatchup(d.matchup||'NN vs NN')}`:d.algorithm==='generational'?`Best population gen ${d.best_population_generation??'—'} · fitness ${d.best_population_fitness==null?'—':Number(d.best_population_fitness).toFixed(3)} · best run ${d.best_run_score==null?'—':Number(d.best_run_score).toFixed(2)}`:d.progress_unit==='episodes'?`AD* hold ${Math.round(100*(d.adstar_defender_hold_rate||0))}% · direct hold ${d.direct_baseline_defender_hold_rate==null?'—':Math.round(100*d.direct_baseline_defender_hold_rate)+'%'} · AD* score ${Math.round(100*(d.adstar_attacker_score_rate||0))}% · score time ${d.adstar_mean_attack_time_to_score==null?'—':d.adstar_mean_attack_time_to_score.toFixed(1)+'s'}`:d.adstar_tracking_error_m==null?'AD* curriculum: awaiting rollout':`AD* error ${d.adstar_tracking_error_m.toFixed(2)}m / ${d.adstar_tracking_target_m||.35}m · imitation ${Number(d.adstar_action_loss_weight||0).toFixed(3)} · task ×${Number(d.task_reward_scale||1).toFixed(2)}`;if(d.algorithm==='crossplay')$('bestPlayback').textContent=`${d.matchup||'NN vs NN'} · seeded robot-on-robot scenarios`;else if(d.algorithm==='generational')$('bestPlayback').textContent=`${trainingTaskLabel(d)} · playback follows saved global best`;else if(d.progress_unit==='episodes')$('bestPlayback').textContent=`FUEL defense evaluation · opponent score rate ${Math.round(100*(d.adstar_attacker_score_rate||0))}%`;else $('bestPlayback').textContent='Saved gamepiece/scoring playback';$('checkpoint').textContent=d.checkpoint||'No saved policy checkpoint yet';$('evaluation').textContent=d.algorithm==='generational'?'Fitness includes gamepiece acquisitions and FUEL scoring against the current curriculum pool.':d.progress_unit==='episodes'?'Gamepiece acquisition/scoring evaluation':'';const elapsed=training.status==='running'&&training.started_at!=null&&Number.isFinite(Number(training.started_at))?Date.now()/1000-Number(training.started_at):Number(training.elapsed_seconds);$('updated').textContent=Number.isFinite(elapsed)?`${elapsedLabel(elapsed)} ${training.status==='running'?'elapsed':'total'}`:'—';const gameRuns=training.game_runs||[];$('trainingEpochs').textContent=gameRuns.length?gameRuns.map(r=>`${r.task==='counter_defense'?'Offense':'Defense'} ${r.current_generation||r.generation||0}/${r.total_generations||0}`).join(' · '):training.generation==null?'—':`${training.generation} / ${training.total_generations||'—'}`;$('trainingRun').textContent=gameRuns.length?gameRuns.map(r=>`${r.task==='counter_defense'?'Offense':'Defense'} ${r.status} · ${Number(r.completed_timesteps||0).toLocaleString()} transitions · ${r.device||''}`).join(' | '):training.run_name?`${training.run_name} · ${training.status||'unknown'}`:'No generational run found';if(d.completed_timesteps!=null){history.push(d.completed_timesteps);history=history.slice(-100);drawChart()}}catch(e){if(revision===selectionRevision)$('state').textContent='Waiting for training status';}
if(revision!==selectionRevision||run!==selectedRun)return;try{if(!$('zoneScenario').value)await refreshPlayback(run,revision)}catch(e){if(revision===selectionRevision)$('matchupNotice').textContent=`Could not load playback: ${e.message}`}if(revision===selectionRevision)schedulePoll()}
async function loadZonePlayback(){const selection=$('zoneScenario').value;if(!selection||!selectedRun||zoneLoading)return;zoneLoading=true;const revision=selectionRevision,run=selectedRun,[start,goal]=selection.split('-');$('matchupNotice').textContent=`Sampling a new ${start} → ${goal} rollout…`;$('zoneScenario').disabled=true;try{const response=await fetch(`/api/zone-playback?run=${encodeURIComponent(run)}&start=${start}&goal=${goal}&seed=${Math.floor(Math.random()*2147483647)}`,{cache:'no-store'});const record=await response.json();if(!response.ok)throw new Error(record.error||`Rollout request failed (${response.status})`);if(revision!==selectionRevision||run!==selectedRun)return;playbackTask=record.task;world=record.field||world;loadedScenarios=record.scenarios||[];const scenario=loadedScenarios[0];frames=scenario?.frames||[];index=0;$('scenario').replaceChildren(new Option(scenario?.label||'Zone rollout',scenario?.id||'zone'));$('scenario').dataset.catalog=`zone:${selection}:${record.seed}`;$('scenario').value=scenario?.id||'zone';$('scenarioCount').textContent=scenario?.label||'';$('showGhosts').checked=false;$('frame').max=Math.max(0,frames.length-1);$('matchupNotice').textContent=`${start[0].toUpperCase()+start.slice(1)} → ${goal[0].toUpperCase()+goal.slice(1)} · fresh start and goal sampled`;drawField()}catch(error){if(revision===selectionRevision)$('matchupNotice').textContent=`Could not create zone rollout: ${error.message}`}finally{zoneLoading=false;if(revision===selectionRevision)$('zoneScenario').disabled=false}}
async function nextPlayback(){if($('fitnessCandidate').value){const [generation,rank]=($('fitnessCandidate').value||'').split(':').map(Number),ranked=loadedScenarios.filter(item=>item.rank).sort((a,b)=>a.rank-b.rank);if(ranked.length>1){const current=ranked.findIndex(item=>item.rank===rank),next=ranked[(current+1)%ranked.length];$('fitnessCandidate').value=`${generation}:${next.rank}`;await displayFitnessCandidate(generation,next.rank)}else{restartPlaybackClock();drawField()}return}if($('zoneScenario').value){await loadZonePlayback();return}if(generationCatalog.length){const latest=String(generationCatalog[generationCatalog.length-1].generation);if($('generation').value!==latest)$('generation').value=latest;await loadGeneration(Number(latest),true);return}if(loadedScenarios.length>1){const current=$('scenario').value,candidates=loadedScenarios.filter(s=>s.id!==current),next=candidates[Math.floor(Math.random()*candidates.length)];$('scenario').value=next.id;frames=next.frames||[];index=0;updateScenarioCount()}else restartPlaybackClock()}
function resetPlayback(){history=[];frames=[];loadedScenarios=[];generationCatalog=[];generationCache={};index=0;playbackRevision=null;$('fitnessInspector').hidden=true;$('fitnessCandidate').replaceChildren();autoBestKey='';$('generation').hidden=true;$('generation').dataset.catalog='';$('scenario').dataset.catalog='';$('scenario').replaceChildren(new Option('Scenario',''));$('scenarioCount').textContent='';drawField()}function selectMatchup(changedRole='',refresh=true){const attacker=$('attackerMode').value,defender=$('defenderMode').value,match=playbackRuns.find(run=>run.attacker===attacker&&run.defender===defender);selectionRevision++;selectedRun=match?.id||'';resetPlayback();$('matchupNotice').textContent=match?'':`No saved playback for ${$('activeStrategy').textContent} vs ${$('activeOpponent').textContent}.`;if(match)$('bestPlayback').textContent=displayMatchup(match.label);if(refresh){clearTimeout(pollTimer);if($('zoneScenario').value)loadZonePlayback();else poll()}}function activeOpponentLabel(role,value){if(value==='mixed')return 'Mixed opponents';if(role==='offense')return ({scripted:'Scripted Defense + AD*',adstar:'Reactive AD* Defense',nn:'NN Strategy + AD*'})[value]||'—';return ({scripted:'Scripted Strategy + AD*',adstar:'AD* Offense',nn:'NN Strategy + AD*'})[value]||'—'}function updateRoleOptions(preserve=true){const role=$('roleSelector').value,controller=$('controllerSelector'),opponent=$('opponentSelector'),oldController=preserve?controller.value:'',oldOpponent=preserve?opponent.value:'';const controllers=role==='offense'?[['scripted','Scripted Strategy'],['nn','NN Strategy']]:[['scripted','Scripted Strategy'],['reactive','Reactive AD* Defense'],['nn','NN Strategy']];controller.replaceChildren(...controllers.map(([value,label])=>new Option(label,value)));controller.value=controllers.some(item=>item[0]===oldController)?oldController:'nn';const opponents=[['scripted','Scripted'],['adstar','AD*'],['nn','NN'],['mixed','Mixed']];opponent.replaceChildren(...opponents.map(([value,label])=>new Option(label,value)));opponent.value=opponents.some(item=>item[0]===oldOpponent)?oldOpponent:'scripted';$('opponentHelp').textContent=role==='defense'?'The deterministic offense baseline uses scripted objective selection with AD* motion.':'Choose which defense strategy the offense will face.';syncRoleSelectors(true)}function syncRoleSelectors(refresh=true){const role=$('roleSelector').value,controller=$('controllerSelector').value,opponent=$('opponentSelector').value,attacker=role==='offense'?(controller==='nn'?'nn':'adstar'):(opponent==='nn'?'nn':'adstar'),defender=role==='defense'?(controller==='nn'?'nn':controller==='reactive'?'adstar_defender':'guard'):(opponent==='nn'?'nn':opponent==='adstar'?'adstar_defender':opponent==='mixed'?'mixed':'guard');$('attackerMode').value=attacker;$('defenderMode').value=defender;$('activeRole').textContent=role==='offense'?'Offense':'Defense';$('activeStrategy').textContent=controller==='nn'?'NN Strategy + AD*':controller==='reactive'?'Reactive AD* Defense':controller==='scripted'?(role==='offense'?'Scripted Strategy + AD*':'Scripted Defense + AD*'):'—';$('activeOpponent').textContent=activeOpponentLabel(role,opponent);selectMatchup(role,refresh)}$('roleSelector').onchange=()=>updateRoleOptions(false);$('controllerSelector').onchange=()=>syncRoleSelectors();$('opponentSelector').onchange=()=>syncRoleSelectors();function syncVisibleSelectors(){const role=$('roleSelector').value,attacker=$('attackerMode').value,defender=$('defenderMode').value;if(role==='offense'){$('controllerSelector').value=attacker==='nn'?'nn':'scripted';$('opponentSelector').value=defender==='nn'?'nn':defender==='adstar_defender'?'adstar':'scripted'}else{$('controllerSelector').value=defender==='nn'?'nn':defender==='adstar_defender'?'reactive':'scripted';$('opponentSelector').value=attacker==='nn'?'nn':'adstar'}$('activeRole').textContent=role==='offense'?'Offense':'Defense';$('activeStrategy').textContent=$('controllerSelector').selectedOptions[0]?.textContent||'—';if($('activeStrategy').textContent==='NN Strategy')$('activeStrategy').textContent='NN Strategy + AD*';if($('activeStrategy').textContent==='Scripted Strategy')$('activeStrategy').textContent=role==='offense'?'Scripted Strategy + AD*':'Scripted Defense + AD*';$('activeOpponent').textContent=activeOpponentLabel(role,$('opponentSelector').value)}$('generation').onchange=()=>{index=0;loadGeneration(Number($('generation').value),true)};$('fitnessCandidate').onchange=selectFitnessCandidate;$('playbackSpeed').onchange=e=>{const now=performance.now();if(playing)playbackAnchorSim=playbackAnchorSim+(now-playbackAnchorWall)/1000*playbackSpeed;playbackAnchorWall=now;playbackSpeed=Number(e.target.value)||1};$('scenario').onchange=()=>{const s=loadedScenarios.find(x=>x.id===$('scenario').value);frames=s?.frames||[];index=0;updateScenarioCount();drawField()};$('frame').oninput=e=>{ensurePlaybackClock();index=+e.target.value;playbackSimTime=playbackTimes[index]||0;frameAlpha=0;playbackAnchorSim=playbackSimTime;playbackAnchorWall=performance.now();drawField()};$('frame').addEventListener('pointerdown',()=>{playing=false});$('frame').addEventListener('pointerup',()=>{playbackAnchorSim=playbackSimTime;playbackAnchorWall=performance.now();playing=true});$('frame').addEventListener('pointercancel',()=>{playbackAnchorSim=playbackSimTime;playbackAnchorWall=performance.now();playing=true});$('frame').addEventListener('keydown',()=>{playing=false});$('frame').addEventListener('keyup',()=>{playbackAnchorSim=playbackSimTime;playbackAnchorWall=performance.now();playing=true});$('play').onclick=()=>{const resume=!playing;if(resume&&index>=frames.length-1)nextPlayback();playing=resume;$('play').textContent=playing?'Pause':'Play'};$('zoneScenario').onchange=()=>{if($('zoneScenario').value)loadZonePlayback();else{selectionRevision++;poll()}};async function loadPlaybackRuns(){try{const response=await fetch('/api/playback-runs',{cache:'no-store'});if(!response.ok)return;playbackRuns=await response.json();const requested=playbackRuns.find(run=>run.id===initialRun)||playbackRuns[0];if(requested){$('attackerMode').value=requested.attacker;$('defenderMode').value=requested.defender}updateRoleOptions(true);syncVisibleSelectors();loadGameEvaluations();selectMatchup(false)}catch(e){$('matchupNotice').textContent='Could not load playback options.'}}
function animatePlayback(){if(frames.length&&!zoneLoading){ensurePlaybackClock();if(playing&&!playbackTransition){playbackSimTime=playbackAnchorSim+(performance.now()-playbackAnchorWall)/1000*playbackSpeed;const end=playbackTimes[playbackTimes.length-1]||0;if(playbackSimTime>=end){setPlaybackTime(end);drawField();playbackTransition=true;Promise.resolve(nextPlayback()).finally(()=>{playbackTransition=false;ensurePlaybackClock();drawField()})}else{setPlaybackTime(playbackSimTime);drawField()}}}requestAnimationFrame(animatePlayback)}requestAnimationFrame(animatePlayback);loadPlaybackRuns().then(poll);loadAblations();setInterval(loadAblations,15000);
$('gameEvaluation').onchange=selectGameEvaluation;$('showGrid').onchange=drawField;
$('showGhosts').onchange=drawField;
 $('showFuel').onchange=drawField;
</script></body></html>'''


INFO_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field guide · FRC Training Monitor</title><style>
:root{color-scheme:dark;--bg:#10151b;--card:#19222b;--line:#2c3a47;--muted:#a5b2bd;--green:#68e0b0;--blue:#0066B3;--red:#ED1C24;--orange:#ED1C24;--pink:#f17cda}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf4f7;font:15px/1.6 system-ui,sans-serif}a{color:var(--blue);text-underline-offset:3px}a:focus-visible{outline:2px solid var(--green);outline-offset:3px}.wrap{max-width:900px;margin:auto;padding:28px 20px 56px}header{display:flex;justify-content:space-between;align-items:center;gap:16px;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:32px}h1{font-size:clamp(27px,5vw,40px);line-height:1.12;letter-spacing:-.025em;margin:0}h2{font-size:22px;line-height:1.25;margin:34px 0 8px}h3{font-size:16px;margin:20px 0 4px}p{margin:8px 0;color:#d5dfe5}.lede{font-size:18px;color:#c3d0d8;max-width:68ch}.section{padding-bottom:22px;border-bottom:1px solid var(--line)}.note{color:var(--muted);font-size:13px}.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:16px 0}.legend span{white-space:nowrap}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}.green{background:var(--green)}.blue{background:var(--blue)}.red{background:var(--red)}.orange{background:var(--red)}.pink{background:var(--pink)}code{font: .92em ui-monospace,monospace;color:#c6d8ff;background:#111b23;padding:2px 5px;border-radius:4px}ul{padding-left:22px;color:#d5dfe5}li{padding:3px 0}.back{font-weight:600;white-space:nowrap}
@media(max-width:560px){.wrap{padding:20px 16px 40px}header{align-items:flex-start;flex-direction:column;margin-bottom:24px}h2{margin-top:26px}}
</style></head><body><main class="wrap"><header><h1>Field guide</h1><a class="back" href="/">← Live dashboard</a></header>
<p class="lede">What the training monitor simulates, how its planners work, and what the playback is showing.</p>
<section class="section"><h2>What is being trained</h2><p>The dashboard follows PPO training for FRC gamepiece tasks. Select <strong>Offense</strong> to inspect gamepiece acquisition and scoring, or <strong>Defense</strong> to inspect disruption strategies. Strategy controls choose how objectives are selected; AD* plans the robot's motion.</p><p>Episodes use randomized field state and gamepiece availability. This varies the approach direction and reduces dependence on one fixed route.</p></section>
<section class="section"><h2>Field and robot model</h2><p>The field layout represents the 2026 REBUILT playing area, including alliance zones, hubs, bumps, trench structures, towers, and depots. Trench crossbars are overhead and passable; only the in-field ends of the side supports block robots. The guardrail-side ends do not add extra collider strips beyond the field boundary. Each bump has a triangular height profile across its 44.4-inch depth and 6.513-inch peak. Wheel contact heights estimate chassis pitch and load transfer, while ramp grade adds or opposes gravity along the field. AD* retains a smaller conservative traversal cost.</p><p>Robots use four-module swerve kinematics. The gamepiece intake is on the robot's local +X/front side; pickup only occurs in a short capture band ahead of that bumper. Playback marks the intake edge with a gold bar. Drive acceleration comes from motor torque, gearing, current limits, robot mass, and tire grip; module locations, steering response, battery resistance, and current limits are configurable. Contacts use oriented bumper geometry, frictional impulses, and chassis yaw inertia, so off-center impacts can rotate the robot. Training status identifies whether it uses the built-in illustrative values or a robot configuration file.</p><p class="note">The drivetrain is not an IRL digital twin until its configuration is populated from robot hardware and checked against measured drive, brake, steer, and yaw traces. The fast model still approximates steering motor control, tire slip, bumper deformation, suspension, battery chemistry, and other robot loads.</p></section>
<section class="section"><h2>AD* route planning</h2><p>The reference field planner follows PathPlanner LocalADStar. Tensor training uses a batched GPU grid search derived from the same 8-connected, clearance-aware route costs. It solves every world together with parallel potential updates, accounts for a velocity-predicted defender intercept, and keeps route search and tracking tensors on the training device.</p><p>Routes use rotated rectangular bumper clearance, field boundaries, trench supports, bump regions, curvature and steering-rate speed caps, and a braking profile. Training replans on a fixed control-step cadence to avoid device synchronization. Playback shows both obstacle-aware AD* reference routes as dotted lines, tinted to match the attacker and defender.</p></section>
<section class="section"><h2>Read the playback</h2><div class="legend"><span><i class="dot blue"></i>Blue: defender</span><span><i class="dot red"></i>Red: attacker</span><span><i class="dot" style="background:#ff9d91"></i>Warm dotted: attacker AD* route</span><span><i class="dot" style="background:#80caff"></i>Cool dotted: available defender AD* route</span><span>White arrows: commanded effort on both robots</span><span>Yellow diamond: predicted intercept</span><span>Gold dots: FUEL · Hub rings: active or inactive</span></div><p>Robot colors use FIRST alliance red (#ED1C24) and blue (#0066B3). Policy observations use perceived robot and gamepiece state. Deterministic strategies choose objectives from that state, and AD* plans motion. Dotted routes show the available AD* paths. The playback slider scrubs the selected rollout and the opponent selector chooses the matchup when a saved rollout exists.</p></section>
<section><h2>Training monitor terms</h2><ul><li><strong>Progress:</strong> environment timesteps completed against the configured target.</li><li><strong>Throughput:</strong> simulation transitions processed per second; it varies with workload and hardware utilization.</li><li><strong>Accelerator:</strong> device and backend reported by the running training process.</li><li><strong>Checkpoint / test:</strong> latest saved policy checkpoint and available evaluation summary.</li><li><strong>AD* error:</strong> distance between the learned action/trajectory and the planner reference during imitation-guided training.</li></ul><p class="note">Metrics appear only when the selected run has produced the corresponding status or playback data.</p></section>
</main></body></html>'''


def load_ablation_records(run_dir: Path) -> list[dict]:
    """Read canonical ablation reports for the controller comparison."""
    records: list[dict] = []
    canonical_paths = [run_dir.parent / "metrics" / "ablations.json",
                       run_dir.parent.parent / "metrics" / "ablations.json",
                       run_dir / "metrics" / "ablations.json"]
    if run_dir.is_dir():
        canonical_paths.extend(run_dir.glob("*/ablations.json"))
    canonical_paths = list(dict.fromkeys(canonical_paths))
    for path in canonical_paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        evaluations = payload.get("evaluations", []) if isinstance(payload, dict) else []
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            record = dict(evaluation)
            record.setdefault("run_name", "metrics" if path.name == "ablations.json" and path.parent.name == "metrics" else path.parent.name)
            record.setdefault("comparable", False)
            record["source"] = path.name
            records.append(record)

    return records


def create_handler(run_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed=urlparse(self.path)
            params=parse_qs(parsed.query)
            run=params.get("run",[""])[0]
            target_name=run
            target=run_dir/target_name if re.fullmatch(r"[a-z0-9-]+",target_name) else run_dir
            if parsed.path == "/" or parsed.path == "/index.html":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif parsed.path == "/info":
                body = INFO_PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif parsed.path == "/api/training-status":
                runs=[]
                for status_path in run_dir.glob("*/status.json"):
                    try:
                        status=json.loads(status_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if status.get("algorithm") == "generational":
                        status["run_name"] = status_path.parent.name
                        runs.append(status)
                # A killed trainer can leave a status file marked "running".
                # Treat it as live only while the status file is being refreshed.
                running=[status for status in runs if status.get("status") == "running"
                         and status_path_age(run_dir / status["run_name"]) < 45 * 60]
                candidates=running or runs
                if candidates:
                    selected=max(candidates,key=lambda status: float(status.get("started_at") or 0.))
                    game_runs=[{key:status.get(key) for key in (
                        "run_name","task","status","architecture","generation",
                        "total_generations","completed_timesteps","requested_timesteps",
                        "transitions_per_second","device")}
                        for status in candidates if status.get("architecture")=="strategic_adstar"]
                    body=json.dumps({**selected,"game_runs":game_runs}).encode()
                else:
                    body=json.dumps({"status":"waiting","algorithm":"generational"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/playback-runs":
                runs=[]
                for child in run_dir.iterdir():
                    if (child.is_dir() and re.fullmatch(r"[a-z0-9-]+", child.name)
                            and (child / "playback.json").is_file()):
                        try:
                            playback=json.loads((child / "playback.json").read_text())
                            status=json.loads((child / "status.json").read_text())
                        except (OSError, json.JSONDecodeError):
                            continue
                        task=playback.get("task")
                        attacker=defender=None
                        if playback.get("matchup") == "nn-vs-nn":
                            attacker,defender="nn","nn"
                        elif task == "adstar_attacker_defense":
                            attacker,defender="adstar","nn"
                        elif task == "counter_defense" and status.get("opponent") == "adstar_defender":
                            attacker,defender="nn","adstar_defender"
                        elif task == "counter_defense" and status.get("opponent","guard") == "guard":
                            attacker,defender="nn","guard"
                        elif (task == "defense" and status.get("algorithm") == "generational"
                              and "adstar" in status.get("opponents", [])):
                            # Mixed generational playback is recorded against AD*.
                            attacker,defender="adstar","nn"
                        if attacker is None:
                            continue
                        runs.append({"id": child.name,
                            "label": child.name.replace("-", " ").title(),
                            "attacker": attacker, "defender": defender,
                            "mtime": (child / "playback.json").stat().st_mtime})
                runs.sort(key=lambda item: item["mtime"], reverse=True)
                for item in runs:
                    item.pop("mtime", None)
                body=json.dumps(runs).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/ablations":
                body=json.dumps(load_ablation_records(run_dir)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/game-evaluations":
                labels={"scripted_offense_adstar":"Scripted offense + AD*",
                        "learned_offense_adstar":"Learned offense + AD*",
                        "scripted_defense_adstar":"Scripted defense + AD*",
                        "learned_defense_adstar":"Learned defense + AD*"}
                roots=(run_dir.parent / "metrics", run_dir.parent.parent / "metrics", run_dir / "metrics")
                entries=[{"id":name,"label":label} for name,label in labels.items()
                         if any((root / f"ablations-{name}" / "playback.json").is_file() for root in roots)]
                body=json.dumps(entries).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/game-evaluation-playback":
                name=params.get("name",[""])[0]
                allowed={"scripted_offense_adstar","learned_offense_adstar",
                         "scripted_defense_adstar","learned_defense_adstar"}
                if name not in allowed:
                    self.send_error(400,"unknown game evaluation")
                    return
                roots=(run_dir.parent / "metrics", run_dir.parent.parent / "metrics", run_dir / "metrics")
                playback_path=next((root / f"ablations-{name}" / "playback.json" for root in roots
                                    if (root / f"ablations-{name}" / "playback.json").is_file()),None)
                if playback_path is None:
                    self.send_error(404,"game evaluation playback is unavailable")
                    return
                try:
                    record=json.loads(playback_path.read_text())
                    _complete_playback_adstar_paths(record)
                    body=json.dumps(record,separators=(",",":")).encode()
                except (OSError,ValueError,TypeError,KeyError):
                    self.send_error(500,"could not read game evaluation playback")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/status":
                body = _json_or_default(target / "status.json", {"status": "waiting"})
                if target_name == "adstar-attacker-defense":
                    status = json.loads(body)
                    baseline_raw = _json_or_default(target / "direct-baseline-metrics.json", {})
                    baseline = json.loads(baseline_raw)
                    if (baseline.get("checkpoint") == status.get("checkpoint") and
                            (baseline.get("checkpoint_mtime") is None or
                             baseline.get("checkpoint_mtime") == status.get("checkpoint_mtime"))):
                        status["direct_baseline_defender_hold_rate"] = baseline.get("mean_success")
                        status["direct_baseline_mean_attack_time"] = (
                            baseline.get("mean_time_to_goal", 0.) * baseline.get("episodes", 0) /
                            max(1, baseline.get("episodes", 0) - round(baseline.get("mean_success", 0.) * baseline.get("episodes", 0))))
                    body = json.dumps(status).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/zone-playback":
                start_zone=params.get("start",[""])[0]
                goal_zone=params.get("goal",[""])[0]
                seed_text=params.get("seed",[""])[0]
                if start_zone not in ("red","center","blue") or goal_zone not in ("red","center","blue") or start_zone==goal_zone:
                    self.send_error(400,"start and goal must be distinct red, center, or blue zones")
                    return
                try:
                    seed=int(seed_text) if seed_text.isdigit() else secrets.randbits(31)
                    record=run_zone_playback(run_dir,target_name,start_zone,goal_zone,seed)
                    body=json.dumps(record,separators=(",",":")).encode()
                except (OSError,ValueError,RuntimeError,KeyError) as exc:
                    body=json.dumps({"error":str(exc)}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type","application/json")
                    self.send_header("Content-Length",str(len(body)))
                    self.send_header("Cache-Control","no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type","application/json")
            elif parsed.path == "/api/playback":
                playback_path = target / "playback.json"
                body = _json_or_default(playback_path, {"frames": []})
                try:
                    stat = playback_path.stat()
                    revision = f'"{stat.st_mtime_ns}-{stat.st_size}"'
                    if self.headers.get("If-None-Match") == revision:
                        self.send_response(304)
                        self.send_header("ETag", revision)
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        return
                except FileNotFoundError:
                    revision = "missing"
                try:
                    record = json.loads(body)
                    _complete_playback_adstar_paths(record)
                    body = json.dumps(record, separators=(",", ":")).encode()
                except (ValueError, TypeError, KeyError):
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Playback-Revision", revision)
                self.send_header("ETag", revision)
            elif parsed.path == "/api/generation":
                generation = params.get("generation", [""])[0]
                if not generation.isdigit() or int(generation) < 1:
                    self.send_error(400, "generation must be a positive integer")
                    return
                generation_path = target / "generation-playback" / f"generation-{int(generation)}.json"
                body = _json_or_default(generation_path, {"generation": int(generation), "candidates": []})
                # Older top-five generations contain poses but predate per-run
                # AD* route recording. Supply a real field-aware route for each
                # candidate so the UI can render planner context for every ghost.
                try:
                    record = json.loads(body)
                    if record.get("candidates"):
                        _complete_candidate_adstar_paths(record)
                        body = json.dumps(record, separators=(",", ":")).encode()
                except (ValueError, TypeError, KeyError):
                    pass
                try:
                    stat = generation_path.stat()
                    revision = f'"{stat.st_mtime_ns}-{stat.st_size}"'
                    if self.headers.get("If-None-Match") == revision:
                        self.send_response(304)
                        self.send_header("ETag", revision)
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        return
                except FileNotFoundError:
                    revision = "missing"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("ETag", revision)
            else:
                self.send_error(404)
                return
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass
    return Handler


def run_zone_playback(run_dir: Path, run_name: str, start_zone: str,
                      goal_zone: str, seed: int) -> dict:
    """Use the dashboard runtime when possible, otherwise the project venv."""
    try:
        import torch  # noqa: F401
    except ImportError:
        interpreter=Path.cwd()/".venv"/"bin"/"python"
        if not interpreter.is_file():
            raise RuntimeError("PyTorch is unavailable and the project virtual environment was not found")
        source=("import json,sys; from pathlib import Path; from frc_defense.dashboard import generate_zone_playback; "
                "print(json.dumps(generate_zone_playback(Path(sys.argv[1]),sys.argv[2],sys.argv[3],"
                "sys.argv[4],int(sys.argv[5])),separators=(',',':')))" )
        result=subprocess.run([str(interpreter),"-c",source,str(run_dir.resolve()),run_name,
            start_zone,goal_zone,str(seed)],cwd=Path.cwd(),capture_output=True,text=True,timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[-1200:] or "zone rollout process failed")
        return json.loads(result.stdout)
    return generate_zone_playback(run_dir,run_name,start_zone,goal_zone,seed)


def generate_zone_playback(run_dir: Path, run_name: str, start_zone: str,
                           goal_zone: str, seed: int) -> dict:
    """Run one fresh, zone-constrained sample for a saved dashboard matchup."""
    import torch
    from .tensor_sim import TensorDefenseEnv
    from .tensor_training import (ActorCritic, OBS_DIM, OBS_NORMALIZATION,
        STRATEGIC_ACTION_DIM, _legacy_strategic_action, _reset_obs)
    from .observation import normalize_tensor_observation_batch_in_place

    if not re.fullmatch(r"[a-z0-9-]+",run_name):
        raise ValueError("unknown playback run")
    source=run_dir/run_name
    status=json.loads((source/"status.json").read_text())
    playback=json.loads((source/"playback.json").read_text())
    task=playback.get("task","counter_defense")
    sim_task="defense" if task=="adstar_attacker_defense" else task
    crossplay=playback.get("matchup")=="nn-vs-nn"
    device=torch.device("cpu")

    def load_model(checkpoint):
        path=Path(checkpoint)
        if not path.is_absolute():
            path=(Path.cwd()/path).resolve()
        if not path.is_file():
            raise ValueError(f"policy checkpoint is missing: {path}")
        payload=torch.load(path,map_location=device,weights_only=True)
        model=ActorCritic(payload.get("obs_dim",OBS_DIM),payload.get("action_dim",3)).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model,payload

    if crossplay:
        defender,defender_data=load_model(status.get("defender_checkpoint") or playback.get("defender_checkpoint"))
        attacker,attacker_data=load_model(status.get("attacker_checkpoint") or playback.get("attacker_checkpoint"))
        normalize=defender_data.get("observation_normalization")==OBS_NORMALIZATION
        if normalize!=(attacker_data.get("observation_normalization")==OBS_NORMALIZATION):
            raise ValueError("NN matchup policies use different observation normalization")
        opponent="adstar"
        learned_model=None
    else:
        checkpoint=status.get("checkpoint")
        if not checkpoint:
            raise ValueError("selected matchup has no policy checkpoint")
        learned_model,payload=load_model(checkpoint)
        normalize=payload.get("observation_normalization")==OBS_NORMALIZATION
        opponent=status.get("opponent","adstar" if task=="defense" else "guard")
        if task=="defense" and opponent in ("mixed","guard"):
            opponent="adstar"

    env=TensorDefenseEnv(num_envs=1,task=sim_task,device=device,seed=seed,
        opponent=opponent,horizon=8000,normalize_observations=normalize)
    env.randomize=True
    obs=_reset_obs(env.reset(seed=seed,options={"start_zone":start_zone,"goal_zone":goal_zone}))
    frames=[]
    frame_stride=max(1,env.horizon//500)
    with torch.no_grad():
        for step in range(env.horizon):
            if crossplay:
                obs_defender=env._obs()
                sim=env.sim
                own,other=1,0
                relative=env.goal-sim.pose[:,own,:2]
                parts=(sim.pose[:,own],sim.velocity[:,own],sim.pose[:,other],sim.velocity[:,other],
                    relative,env.goal_radius[:,None],
                    torch.full((1,1),sim.field_length),torch.full((1,1),sim.field_width),
                    sim.length[:,[own,other]],sim.width[:,[own,other]],sim.accel[:,[own,other]]/10.)
                obstacles=(torch.cat((sim.obstacles,env.field_feature_obstacles),0)
                    if env.field_feature_obstacles.shape[0] else sim.obstacles)
                features=torch.zeros((1,12))
                if obstacles.shape[0]:
                    distance=(obstacles[None,:,:2]-sim.pose[:,own,None,:2]).square().sum(-1)
                    chosen=obstacles[distance.topk(min(4,obstacles.shape[0]),dim=-1,largest=False).indices]
                    features[:,:chosen.shape[1]*3]=chosen.reshape(1,-1)
                obs_attacker=torch.cat(parts+(features,),-1)
                if env.normalize_observations:
                    normalize_tensor_observation_batch_in_place(obs_attacker,sim.field_length,
                        sim.field_width,sim.speed[:,[own,other]],sim.omega_limit[:,[own,other]])
                action_d=defender.sample(obs_defender,deterministic=True)[0]
                action_a=attacker.sample(obs_attacker,deterministic=True)[0]
                command_d=torch.cat((action_d[:,:2]*env.sim.speed[:,0,None],
                    (action_d[:,2]*env.sim.omega_limit[:,0])[:,None]),-1)
                command_a=torch.cat((action_a[:,:2]*env.sim.speed[:,1,None],
                    (action_a[:,2]*env.sim.omega_limit[:,1])[:,None]),-1)
                env.sim.step(torch.stack((command_d,command_a),1))
                done=(env.goal-env.sim.pose[:,1,:2]).norm(dim=-1)<env.goal_radius
                effort=[command_d[0,:2].tolist(),command_a[0,:2].tolist()]
                info={}
            else:
                model_obs=obs[:,:learned_model.trunk[0].in_features]
                action_mask=(model_obs[:,-STRATEGIC_ACTION_DIM:].bool()
                    if learned_model.actor.out_features==STRATEGIC_ACTION_DIM else None)
                action=learned_model.sample(model_obs,deterministic=True,action_mask=action_mask)[0]
                if env.action_mode=="strategic" and learned_model.actor.out_features==3:
                    action=_legacy_strategic_action(action,sim_task)
                obs,_,done,truncated,info=env.step(action)
                done=torch.as_tensor(done).reshape(1)|torch.as_tensor(truncated).reshape(1)
                opponent_effort=info.get("opponent_effort_vector",torch.zeros((1,2)))
                own_effort=action[0,:2]*env.sim.speed[0,0]
                if sim_task=="counter_defense":
                    effort=[own_effort.tolist(),opponent_effort[0].tolist()]
                else:
                    effort=[opponent_effort[0].tolist(),own_effort.tolist()]
            if step%frame_stride==0 or bool(done[0].item()):
                pose=env.sim.pose[0].detach().cpu().tolist()
                sizes=torch.stack((env.sim.length[0],env.sim.width[0]),-1).reshape(-1).detach().cpu().tolist()
                frame={"robots":pose,"goal":env.goal[0].detach().cpu().tolist(),"sizes":sizes,
                    "goal_radius":float(env.goal_radius[0].item()),"robot_effort_vectors":effort}
                paths=info.get("adstar_paths") if isinstance(info,dict) else None
                if paths is not None and len(paths):
                    lengths=info.get("adstar_path_lengths")
                    count=int(lengths[0].item()) if lengths is not None else paths.shape[1]
                    frame["adstar_paths"]=[paths[0,:count].detach().cpu().tolist()]
                frames.append(frame)
            if bool(done[0].item()):
                break
    record={"task":task,"matchup":playback.get("matchup"),"seed":seed,
        "dt":env.dt,"field":playback.get("field",{}),
        "scenarios":[{"id":"zone","label":f"{start_zone.title()} → {goal_zone.title()}",
            "start":env.sim.pose[0,0 if sim_task=="counter_defense" else 1,:2].detach().cpu().tolist(),
            "goal":env.goal[0].detach().cpu().tolist(),"start_zone":start_zone,
            "goal_zone":goal_zone,"frames":frames}]}
    _complete_playback_adstar_paths(record)
    return record


def _complete_candidate_adstar_paths(record: dict) -> None:
    """Attach two role-aware AD* paths to every recorded candidate frame."""
    for candidate in record.get("candidates", []):
        _complete_playback_adstar_paths({
            "task": record.get("task"), "field": record.get("field"),
            "frames": candidate.get("frames") or []})


def _complete_playback_adstar_paths(record: dict) -> None:
    """Fill missing per-robot AD* overlays while retaining recorded live paths."""
    from .adstar import ADStarPlanner
    from .field import bump_boxes, rebuilt_field, static_collision_boxes

    field = record.get("field") or {}
    length = float(field.get("length", 16.54))
    width = float(field.get("width", 8.07))
    boxes = rebuilt_field(length, width)
    colliders = static_collision_boxes(boxes)
    bumps = bump_boxes(boxes)
    task = record.get("task", "counter_defense")
    defense_task = task in ("defense", "adstar_attacker_defense")
    attacker_index = 1 if defense_task else 0
    defender_index = 1 - attacker_index
    frame_groups = [scenario.get("frames") or [] for scenario in record.get("scenarios", [])]
    if not frame_groups:
        frame_groups = [record.get("frames") or []]
    for frames in frame_groups:
        if not frames:
            continue
        first = frames[0]
        robots, goal = first.get("robots") or [], first.get("goal")
        if len(robots) < 2 or not goal or len(goal) < 2:
            continue
        sizes = first.get("sizes") or []
        attacker_start = robots[attacker_index][:2]
        reference_routes = [None, None]
        route_targets = [(attacker_index, goal)]
        if not defense_task:
            # The guard reference follows only the attacker's observed position;
            # never derive its overlay from the scoring goal.
            route_targets.append((defender_index, attacker_start))
        for robot_index, target in route_targets:
            robot = robots[robot_index]
            size_offset = robot_index * 2
            robot_length = sizes[size_offset] if len(sizes) > size_offset else .9
            robot_width = sizes[size_offset + 1] if len(sizes) > size_offset + 1 else .9
            planner = ADStarPlanner(length, width, colliders, bumps,
                robot_length=robot_length, robot_width=robot_width,
                robot_heading=robot[2] if len(robot) > 2 else 0.)
            reference_routes[robot_index] = [list(point) for point in planner.plan(robot[:2], target)]
        for frame in frames:
            paths = frame.get("adstar_paths")
            if not isinstance(paths, list) or len(paths) != 2:
                paths = [reference_routes[0], reference_routes[1]]
            else:
                paths = [paths[i] if isinstance(paths[i], list) and len(paths[i]) > 1
                         else reference_routes[i] for i in range(2)]
            if defense_task:
                paths[defender_index] = []
            frame["adstar_paths"] = paths


def _json_or_default(path: Path, default: dict) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return json.dumps(default).encode()


def status_path_age(run_dir: Path) -> float:
    """Return age of the status file, or infinity when it is missing."""
    try:
        return max(0., time.time() - (run_dir / "status.json").stat().st_mtime)
    except OSError:
        return float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="checkpoints/tensor-ppo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), create_handler(Path(args.run_dir)))
    print(f"FRC training dashboard: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
