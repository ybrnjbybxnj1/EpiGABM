from __future__ import annotations

import json

import streamlit.components.v1 as components

def render_animated_map(
    all_snapshots: dict[int, list[dict]],
    households: dict[int, tuple[float, float]],
    day_results: list[dict],
    initial_day: int = 1,
    height: int = 500,
) -> None:
    days_data = {}
    agent_diary: dict[int, list] = {}
    for day, agents in sorted(all_snapshots.items()):
        buildings = {}
        for a in agents:
            hh_id = a["sp_hh_id"]
            if hh_id not in buildings:
                coords = households.get(hh_id, (0, 0))
                buildings[hh_id] = {
                    "lat": coords[0], "lon": coords[1],
                    "total": 0, "infected": 0, "agents": [],
                }
            buildings[hh_id]["total"] += 1
            if a["state"] == "I":
                buildings[hh_id]["infected"] += 1
            if a["state"] == "E":
                buildings[hh_id].setdefault("exposed", 0)
                buildings[hh_id]["exposed"] += 1
            buildings[hh_id]["agents"].append({
                "id": a["sp_id"],
                "arch": a["archetype"],
                "state": a["state"],
                "ill": a.get("illness_day", 0),
                "iso": a.get("will_isolate", False),
                "mask": a.get("wears_mask", False),
                "wid": a.get("work_id", "X"),
                "narr": a.get("narrative", ""),
            })
            aid = a["sp_id"]
            if aid not in agent_diary:
                agent_diary[aid] = []
            agent_diary[aid].append({
                "d": day, "s": a["state"],
                "n": a.get("narrative", ""),
            })
        days_data[day] = list(buildings.values())
    seir_data = []
    for dr in day_results:
        strains = list(dr["S"].keys())
        main = strains[0] if strains else "H1N1"
        seir_data.append({
            "day": dr["day"],
            "S": dr["S"].get(main, 0),
            "E": dr["E"].get(main, 0),
            "I": dr["I"].get(main, 0),
            "R": dr["R"].get(main, 0),
            "iso": dr.get("n_isolating", 0),
            "mask": dr.get("n_masked", 0),
        })
    sorted_days = sorted(all_snapshots.keys())
    min_day = sorted_days[0] if sorted_days else 1
    max_day = sorted_days[-1] if sorted_days else 1
    html = _build_html(days_data, seir_data, agent_diary, min_day, max_day, initial_day)
    components.html(html, height=height + 50, scrolling=False)

def _build_html(
    days_data: dict,
    seir_data: list,
    agent_diary: dict,
    min_day: int,
    max_day: int,
    initial_day: int,
) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body {{ margin:0;padding:0;height:100%;overflow:hidden;font-family:-apple-system,sans-serif; }}
  *{{ box-sizing:border-box; }}
  #main {{ display:flex; height:100%; }}
  #left {{ flex:1; display:flex; flex-direction:column; }}
  #right {{ width:300px; background:#1a1a1a; color:#ccc; overflow-y:auto; border-left:1px solid #333;
    font-size:12px; padding:0; }}
  #map {{ flex:1; min-height:0; }}
  #controls {{ padding:6px 10px; background:#1e1e1e; display:flex; align-items:center; gap:8px; }}
  #controls button {{ background:#333; color:#fff; border:1px solid #555; border-radius:4px;
    padding:3px 10px; cursor:pointer; font-size:13px; }}
  #controls button:hover {{ background:#444; }}
  #controls button.active {{ background:#2196f3; border-color:#2196f3; }}
  #slider {{ flex:1; accent-color:#2196f3; }}
  #day-label {{ color:#fff; min-width:60px; text-align:center; font-weight:bold; font-size:13px; }}
  #stats {{ padding:4px 10px; background:#222; color:#aaa; display:flex; gap:14px; font-size:12px;
    flex-wrap:wrap; }}
  .sv {{ color:#fff; font-weight:bold; }}
  #chart {{ height:140px; background:#1a1a1a; }}
  #news {{ padding:6px 10px; background:#1e1e1e; color:#ccc; font-size:11px; max-height:40px;
    overflow:hidden; border-top:1px solid #333; }}
  canvas {{ width:100%; height:100%; display:block; }}
  /* right panel */
  .rh {{ padding:10px; background:#252525; font-weight:bold; font-size:13px; color:#fff;
    border-bottom:1px solid #333; position:sticky; top:0; z-index:1; }}
  .agent-card {{ padding:8px 10px; border-bottom:1px solid #2a2a2a; cursor:pointer; }}
  .agent-card:hover {{ background:#252525; }}
  .agent-card.selected {{ background:#1a2a3a; border-left:3px solid #2196f3; }}
  .ac-name {{ font-weight:bold; color:#fff; }}
  .ac-state {{ display:inline-block; padding:1px 6px; border-radius:3px; font-size:10px; color:#fff; }}
  .diary {{ padding:10px; }}
  .diary-entry {{ border-left:3px solid #555; padding:3px 8px; margin:4px 0; }}
  .de-day {{ font-weight:bold; font-size:11px; }}
  .de-narr {{ color:#999; font-style:italic; font-size:11px; margin-top:2px; }}
  .tag {{ display:inline-block; padding:1px 5px; border-radius:3px; font-size:10px;
    margin-right:3px; background:#333; }}
  .hint {{ padding:20px; color:#666; text-align:center; }}
  .contacts {{ padding:6px 10px; font-size:11px; }}
  .contacts-title {{ color:#888; font-size:10px; margin-top:6px; }}
  .contact-badge {{ display:inline-block; padding:1px 5px; border-radius:3px; font-size:10px;
    margin:1px 2px; }}
</style>
</head>
<body>
<div id="main">
<div id="left">
  <div id="map"></div>
  <div id="controls">
    <button id="btn-play" onclick="togglePlay()">&#9654;</button>
    <button onclick="stepDay(-1)">&lt;</button>
    <input type="range" id="slider" min="{min_day}" max="{max_day}" value="{initial_day}">
    <button onclick="stepDay(1)">&gt;</button>
    <span id="day-label">day {initial_day}</span>
    <select id="speed" style="background:#333;color:#fff;border:1px solid #555;border-radius:3px;
      padding:1px;font-size:12px;">
      <option value="200">fast</option><option value="500" selected>mid</option>
      <option value="1000">slow</option>
    </select>
  </div>
  <div id="stats"></div>
  <div id="news"></div>
  <div id="chart"><canvas id="cv"></canvas></div>
</div>
<div id="right">
  <div class="rh" id="panel-title">click a house on the map
    <span id="panel-close" onclick="closePanel()" style="float:right;cursor:pointer;color:#888;display:none;">&times;</span>
  </div>
  <div id="panel-body"><div class="hint">select a building to see its residents and their diary</div></div>
</div>
</div>

<script>
const DD = {json.dumps(days_data)};
const SEIR = {json.dumps(seir_data)};
const DIARY = {json.dumps(agent_diary)};
const MN={min_day}, MX={max_day};
const SC = {{S:'#4caf50',E:'#ffc107',I:'#e60000',R:'#9e9e9e'}};

const map = L.map('map',{{zoomControl:true,attributionControl:false}}).setView([59.944,30.265],13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  maxZoom:19}}).addTo(map);

let markers=L.layerGroup().addTo(map), cur={initial_day}, playing=false, timer=null;
let selAgent=null;

// build workplace index: wid -> [agent ids] (from day 1 data)
const WP={{}};
(DD[MN]||[]).forEach(b=>b.agents.forEach(a=>{{
  if(a.wid && a.wid!=='X'){{
    if(!WP[a.wid]) WP[a.wid]=[];
    WP[a.wid].push(a.id);
  }}
}}));

let selBuildingLatLon=null;
function updMap(day) {{
  markers.clearLayers();
  (DD[day]||[]).forEach(b => {{
    if(!b.lat) return;
    const expo=b.exposed||0;
    const hasI=b.infected>0, hasE=expo>0, hi=hasI||hasE;
    // highlight selected agent's building in blue
    const isSelected=selAgent && b.agents.some(a=>a.id===selAgent);
    // red=infectious, orange=mixed, yellow=exposed only, grey=healthy
    let col='#bbb';
    if(isSelected) col='#2196f3';
    else if(hasI) {{ const pct=b.infected/b.total; col=pct>0.3?'#e60000':pct>0.1?'#ff8c00':'#e60000'; }}
    else if(hasE) col='#ffc107';
    const c=L.circleMarker([b.lat,b.lon],{{
      radius:isSelected?10:hi?Math.max(7,Math.min(b.total*2,16)):5,
      color:col, weight:isSelected?3:hi?2:0.8, fill:true, fillColor:col,
      fillOpacity:isSelected?0.9:hi?0.85:0.35,
    }});
    const tipParts=[`${{b.total}} people`];
    if(b.infected>0) tipParts.push(`${{b.infected}} infectious`);
    if(expo>0) tipParts.push(`${{expo}} exposed`);
    c.bindTooltip(tipParts.join(', '));
    c.on('click',()=>showBuilding(b));
    markers.addLayer(c);
    // also highlight coworkers' buildings
    if(selAgent) {{
      const agentInB=b.agents.find(a=>a.id===selAgent);
      if(agentInB && agentInB.wid && agentInB.wid!=='X') {{
        selBuildingLatLon=[b.lat,b.lon];
      }}
    }}
  }});
}}

function closePanel() {{
  selAgent=null; selArch=''; lastBuilding=null;
  document.getElementById('panel-title').innerHTML=`click a house on the map`+
    `<span id="panel-close" style="display:none"></span>`;
  document.getElementById('panel-body').innerHTML='<div class="hint">select a building to see its residents and their diary</div>';
  setTimeout(()=>updMap(cur),0); // defer to ensure selAgent is null before redraw
}}

function showBuilding(b) {{
  selAgent=null; selArch=''; lastBuilding=b;
  document.getElementById('panel-title').innerHTML =
    `House - ${{b.total}} people, ${{b.infected}} infected`+
    `<span id="panel-close" onclick="closePanel()" style="float:right;cursor:pointer;color:#888;">&times;</span>`;
  setTimeout(()=>updMap(cur),0); // defer to ensure selAgent is null before redraw
  let h='';
  b.agents.forEach(a => {{
    const col=SC[a.state]||'#fff';
    const sel=selAgent===a.id?'selected':'';
    const flags=[];
    if(a.iso) flags.push('isolating');
    if(a.mask) flags.push('mask');
    h+=`<div class="agent-card ${{sel}}" onclick="selectAgent(${{a.id}},'${{a.arch}}')">`;
    h+=`<span class="ac-name">#${{a.id}}</span> `;
    h+=`<span class="ac-state" style="background:${{col}}">${{a.state}}</span> `;
    h+=`<span style="color:#888">${{a.arch}}</span>`;
    if(flags.length) h+=` <span class="tag">${{flags.join('</span><span class="tag">')}}</span>`;
    if(a.narr) h+=`<div class="de-narr">${{a.narr}}</div>`;
    h+=`</div>`;
  }});
  document.getElementById('panel-body').innerHTML=h;
}}

let lastBuilding=null;
function selectAgent(id, arch) {{
  selAgent=id;
  selArch=arch||selArch;
  const entries=DIARY[id]||[];

  // find agent's workplace from current day data
  let agentWid=null;
  (DD[cur]||[]).forEach(b=>b.agents.forEach(a=>{{
    if(a.id===id) agentWid=a.wid;
  }}));

  // refresh map to move blue highlight to newly selected agent
  updMap(cur);

  let h=`<div style="padding:8px 10px;display:flex;align-items:center;gap:8px;">`+
    `<span onclick="if(lastBuilding)showBuilding(lastBuilding)" style="cursor:pointer;color:#2196f3;font-size:16px;">&#8592;</span>`+
    `<span style="color:#fff;font-weight:bold;font-size:14px;">#${{id}} - ${{arch}}</span>`+
    `<span onclick="closePanel()" style="margin-left:auto;cursor:pointer;color:#888;">&times;</span>`+
    `</div>`;

  // contacts section
  if(agentWid && agentWid!=='X') {{
    const coworkers=(WP[agentWid]||[]).filter(c=>c!==id);
    if(coworkers.length>0) {{
      // count sick coworkers on current day
      let sickCount=0;
      (DD[cur]||[]).forEach(b=>b.agents.forEach(a=>{{
        if(coworkers.includes(a.id) && (a.state==='I'||a.state==='E')) sickCount++;
      }}));
      h+=`<div class="contacts"><div class="contacts-title">WORKPLACE #${{agentWid}} - ${{coworkers.length}} coworkers`;
      if(sickCount>0) h+=` (<span style="color:#f44336">${{sickCount}} sick</span>)`;
      h+=`</div>`;
      const showN=10;
      const renderBadges=(list)=>{{
        let s='';
        list.forEach(cid=>{{
          let cState='?';
          (DD[cur]||[]).forEach(b=>b.agents.forEach(a=>{{if(a.id===cid) cState=a.state;}}));
          const cc=SC[cState]||'#555';
          s+=`<span class="contact-badge" style="background:${{cc}}33;color:${{cc}};border:1px solid ${{cc}}" `+
            `onclick="selectAgent(${{cid}},'')">#${{cid}} ${{cState}}</span>`;
        }});
        return s;
      }};
      h+=`<div id="cw-short">${{renderBadges(coworkers.slice(0,showN))}}`;
      if(coworkers.length>showN) {{
        h+=` <span style="color:#2196f3;cursor:pointer" onclick="document.getElementById('cw-short').style.display='none';document.getElementById('cw-full').style.display='block'">+ ${{coworkers.length-showN}} more</span>`;
      }}
      h+=`</div>`;
      if(coworkers.length>showN) {{
        h+=`<div id="cw-full" style="display:none">${{renderBadges(coworkers)}}`;
        h+=` <span style="color:#2196f3;cursor:pointer" onclick="document.getElementById('cw-full').style.display='none';document.getElementById('cw-short').style.display='block'">collapse</span></div>`;
      }}
      h+=`</div>`;
    }}
  }}

  // diary
  h+=`<div class="diary">`;
  const filtered=entries.filter(e=>e.d<=cur).reverse();
  filtered.forEach(e => {{
    const col=SC[e.s]||'#555';
    h+=`<div class="diary-entry" style="border-color:${{col}}">`;
    h+=`<div class="de-day" style="color:${{col}}">Day ${{e.d}} - ${{e.s}}</div>`;
    if(e.n) h+=`<div class="de-narr">${{e.n}}</div>`;
    h+=`</div>`;
  }});
  h+=`</div>`;
  document.getElementById('panel-body').innerHTML=h;
}}

function updStats(day) {{
  const s=SEIR.find(d=>d.day===day);
  if(!s) return;
  document.getElementById('stats').innerHTML=
    `<span>S:<span class="sv" style="color:#4caf50">${{s.S}}</span></span>`+
    `<span>E:<span class="sv" style="color:#ffc107">${{s.E}}</span></span>`+
    `<span>I:<span class="sv" style="color:#f44336">${{s.I}}</span></span>`+
    `<span>R:<span class="sv" style="color:#9e9e9e">${{s.R}}</span></span>`+
    `<span>iso:<span class="sv">${{s.iso}}</span></span>`+
    `<span>mask:<span class="sv">${{s.mask}}</span></span>`;

  // news ticker - epidemic events with realistic thresholds
  const prev=day>MN ? SEIR.find(d=>d.day===day-1) : null;
  let news='';
  const pop=s.S+s.E+s.I+s.R;
  const pct=pop>0?(s.I/pop*100).toFixed(1):0;
  const pctN=parseFloat(pct);
  // news only when epidemic crosses 5% threshold (Russian epidemic threshold, PMC4639464)
  if(s.I===0 && prev && prev.I>0) news='No active cases. The epidemic appears to be over.';
  else if(pctN<5) news='';
  else if(prev && prev.I<pop*0.05 && s.I>=pop*0.05) news=`Epidemic threshold exceeded: ${{s.I}} cases (${{pct}}%). Authorities recommend caution.`;
  else if(prev && s.I>prev.I*1.3 && pctN>10) news=`Flu cases surging: ${{s.I}} infected (${{pct}}%). Clinics overwhelmed.`;
  else if(pctN>15) news=`Epidemic peak: ${{pct}}% of population infected.`;
  else if(prev && s.I<prev.I && pctN>5) news=`Flu declining: ${{s.I}} active cases (${{pct}}%).`;
  else if(pctN>=5) news=`${{s.I}} active flu cases (${{pct}}%).`;
  else news='';
  document.getElementById('news').innerHTML=news?'<b>NEWS:</b> '+news:'';
}}

function drawChart(hd) {{
  const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
  cv.width=cv.offsetWidth;cv.height=cv.offsetHeight;
  const W=cv.width,H=cv.height;
  ctx.fillStyle='#1a1a1a';ctx.fillRect(0,0,W,H);
  if(!SEIR.length) return;
  const p={{l:35,r:8,t:8,b:20}},cw=W-p.l-p.r,ch=H-p.t-p.b;
  // X axis spans full simulation range (MN to MX from snapshots)
  const lastSeirDay=SEIR[SEIR.length-1].day;
  const chartMax=Math.max(MX, lastSeirDay);
  const xf=d=>p.l+((d-MN)/(chartMax-MN||1))*cw;
  // draw only I(t) curve - other compartments hide the peak
  const mx2=Math.max(...SEIR.map(d=>d.I))||1;
  const yf2=v=>p.t+ch-(v/mx2)*ch;
  ctx.beginPath();ctx.strokeStyle='#f44336';ctx.lineWidth=2.5;
  SEIR.forEach((d,i)=>{{const px=xf(d.day),py=yf2(d.I);i?ctx.lineTo(px,py):ctx.moveTo(px,py);}});
  ctx.stroke();
  // fill under curve
  ctx.lineTo(xf(SEIR[SEIR.length-1].day),p.t+ch);ctx.lineTo(xf(SEIR[0].day),p.t+ch);
  ctx.fillStyle='rgba(244,67,54,0.15)';ctx.fill();
  if(hd>=MN){{
    ctx.beginPath();ctx.strokeStyle='rgba(255,255,255,0.5)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
    const dx=xf(hd);ctx.moveTo(dx,p.t);ctx.lineTo(dx,p.t+ch);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#fff';ctx.font='11px sans-serif';ctx.fillText('day '+hd,dx+3,p.t+12);
  }}
  ctx.fillStyle='#555';ctx.font='10px sans-serif';
  ctx.fillText('0',p.l-14,p.t+ch);ctx.fillText(mx2,p.l-30,p.t+10);
  ctx.fillText('I(t)',p.l+4,p.t+14);
}}

const sl=document.getElementById('slider'),dl=document.getElementById('day-label');

let selArch='';
function setDay(d) {{
  cur=Math.max(MN,Math.min(MX,d));
  sl.value=cur; dl.textContent='day '+cur;
  updMap(cur); updStats(cur); drawChart(cur);
  // refresh agent diary if one is selected
  if(selAgent) selectAgent(selAgent, selArch);
}}
sl.addEventListener('input',e=>setDay(parseInt(e.target.value)));
function stepDay(n){{setDay(cur+n);}}

function togglePlay(){{
  playing=!playing;
  const b=document.getElementById('btn-play');
  if(playing){{b.textContent='⏹';b.classList.add('active');pLoop();}}
  else{{b.textContent='▶';b.classList.remove('active');clearTimeout(timer);}}
}}
function pLoop(){{
  if(!playing)return;
  if(cur<MX){{setDay(cur+1);timer=setTimeout(pLoop,parseInt(document.getElementById('speed').value));}}
  else{{playing=false;document.getElementById('btn-play').textContent='▶';
    document.getElementById('btn-play').classList.remove('active');}}
}}

setTimeout(()=>{{map.invalidateSize();setDay({initial_day});}},100);
window.addEventListener('resize',()=>{{map.invalidateSize();drawChart(cur);}});
</script>
</body>
</html>"""
