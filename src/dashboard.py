# Copyright (c) 2026 Abigail Da Costa
# Licensed under the MIT License — see LICENSE file for details.

"""
dashboard.py — Oil Risk Intelligence Platform
Run with: streamlit run dashboard.py
"""
import os
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import plotly.graph_objects as go
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model import train_model
from live_data import get_live_oil
from location_extractor import extract_country

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------

st.set_page_config(
    page_title="Oil Risk Intelligence",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------
# BLOOMBERG-STYLE CSS
# -----------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #05080f !important;
    color: #c8dff0 !important;
}
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #c8dff0 !important; }
.stMetric { background: #0d1a2e; border-radius: 6px; padding: 12px 16px; }
.stMetric label { font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important; color: #4a6a8a !important; }
.stMetric [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 22px !important; color: #c8dff0 !important; }
.stMetric [data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; }
.stDataFrame { background: #0d1a2e !important; border-radius: 6px; }
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #4a6a8a; border-bottom: 0.5px solid #1a3a6a;
    padding-bottom: 6px; margin-bottom: 12px; margin-top: 20px;
}
.stSlider label { font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: #4a6a8a !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# AUTO REFRESH (60s)
# -----------------------------------------------------------------------

st_autorefresh(interval=60_000, key="dashboard_refresh")

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------

st.markdown("""
<div style="display:flex; align-items:baseline; justify-content:space-between; margin-bottom:4px;">
  <div>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:18px; font-weight:500;
      letter-spacing:0.06em; color:#c8dff0;">OIL RISK INTELLIGENCE PLATFORM</span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:11px;
      color:#4a6a8a; margin-left:14px;">REAL-TIME GEOPOLITICAL + MARKET FEED</span>
  </div>
</div>
<div style="height:0.5px; background:#1a3a6a; margin-bottom:18px;"></div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# SHARED PLOTLY LAYOUT
# -----------------------------------------------------------------------

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#05080f",
    plot_bgcolor="#0d1a2e",
    font=dict(family="IBM Plex Mono, monospace", size=11, color="#7aabdf"),
    margin=dict(l=40, r=20, t=36, b=36),
    hovermode="x unified",
    xaxis=dict(gridcolor="#1a3a6a", linecolor="#1a3a6a"),
    yaxis=dict(gridcolor="#1a3a6a", linecolor="#1a3a6a"),
)

# -----------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_oil() -> pd.DataFrame:
    oil = pd.read_csv("data/oil_prices_processed.csv")

    # Drop any duplicate columns (e.g. two 'Date' columns from yfinance CSV format)
    oil = oil.loc[:, ~oil.columns.duplicated()]

    # If Date ended up as the index, pull it out as a column
    if "Date" not in oil.columns:
        oil = oil.reset_index().rename(columns={"index": "Date"})

    oil["Close"] = pd.to_numeric(oil["Close"], errors="coerce")
    oil["volatility"] = pd.to_numeric(oil["volatility"], errors="coerce")
    return oil.dropna(subset=["Close", "volatility"])

@st.cache_data(ttl=300)
def load_news() -> pd.DataFrame:
    news = pd.read_csv("data/news_with_sentiment.csv")
    news["country"] = news["title"].apply(extract_country)
    return news.dropna(subset=["country"])


@st.cache_data(ttl=300)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv("data/ml_dataset.csv")


@st.cache_resource
def get_model():
    """Cache the trained model — avoids retraining on every page refresh."""
    dataset = load_dataset()
    return train_model(dataset)


oil = load_oil()
news = load_news()
model = get_model()

# -----------------------------------------------------------------------
# KPI BAR
# -----------------------------------------------------------------------

latest_price = oil["Close"].iloc[-1]
prev_price = oil["Close"].iloc[-2]
latest_vol = oil["volatility"].iloc[-1]
prev_vol = oil["volatility"].iloc[-2]
avg_sentiment = news["sentiment"].mean()
total_news = len(news)

st.markdown('<div class="section-label">Market Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("WTI Crude (USD)", f"${latest_price:.2f}", f"{latest_price - prev_price:+.2f}")
c2.metric("Volatility (5d σ)", f"{latest_vol:.4f}", f"{latest_vol - prev_vol:+.4f}")
c3.metric("Avg Sentiment", f"{avg_sentiment:.3f}", delta=None)
c4.metric("Articles (total)", f"{total_news:,}")
c5.metric("Active Hotspots", "11")

# -----------------------------------------------------------------------
# LIVE PRICE + VOLATILITY CHARTS
# -----------------------------------------------------------------------

st.markdown('<div class="section-label">Price & Volatility</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    oil_live = get_live_oil()
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=oil_live["Datetime"],
        y=oil_live["Close"],
        mode="lines",
        line=dict(color="#1D9E75", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(29,158,117,0.06)",
        name="WTI Price",
    ))
    fig_price.update_layout(title="Crude Oil Price — Live (1h intervals)", **CHART_LAYOUT)
    st.plotly_chart(fig_price, use_container_width=True)

with col_right:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=oil["Date"],
        y=oil["volatility"],
        mode="lines",
        line=dict(color="#D85A30", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(216,90,48,0.06)",
        name="Volatility",
    ))
    fig_vol.update_layout(title="Rolling Volatility", **CHART_LAYOUT)
    st.plotly_chart(fig_vol, use_container_width=True)


# -----------------------------------------------------------------------
# SENTIMENT MAP + NEWS TABLE
# -----------------------------------------------------------------------

st.markdown('<div class="section-label">Sentiment & News Feed</div>', unsafe_allow_html=True)

col_map, col_news = st.columns([3, 2])

with col_map:
    country_sentiment = news.groupby("country")["sentiment"].mean().reset_index()
    fig_map = go.Figure(go.Choropleth(
        locations=country_sentiment["country"],
        locationmode="country names",
        z=country_sentiment["sentiment"],
        colorscale="RdYlGn",
        reversescale=True,
        colorbar=dict(
            title=dict(text="Sentiment", font=dict(family="IBM Plex Mono", size=11)),
            tickfont=dict(family="IBM Plex Mono", size=10),
            len=0.6,
        ),
    ))
    fig_map.update_layout(
        title="Global Oil Market Sentiment",
        geo=dict(
            bgcolor="#05080f",
            lakecolor="#0d1e35",
            landcolor="#142236",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#1a3a6a",
        ),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "hovermode")},
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_news:
    st.markdown("**Latest Geopolitical News**")
    st.dataframe(
        news[["title", "country", "sentiment"]].head(20),
        use_container_width=True,
        height=360,
    )

# -----------------------------------------------------------------------
# AI VOLATILITY PREDICTOR + EVENTS
# -----------------------------------------------------------------------

st.markdown('<div class="section-label">AI Predictor & Event Alerts</div>', unsafe_allow_html=True)

col_pred, col_events = st.columns(2)

with col_pred:
    news_count = st.slider("News Volume (articles)", 0, 20, 5)
    sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, 0.0, step=0.01)
    prediction = model.predict([[news_count, sentiment_score]])[0]

    level = "CRITICAL" if prediction > 0.03 else "ELEVATED" if prediction > 0.02 else "NORMAL"
    level_color = "#D85A30" if prediction > 0.03 else "#EF9F27" if prediction > 0.02 else "#1D9E75"

    st.markdown(f"""
    <div style="background:#0d1a2e; border-radius:8px; padding:16px 20px; margin-top:8px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:0.1em;
        text-transform:uppercase; color:#4a6a8a;">Predicted Volatility</div>
      <div style="display:flex; align-items:baseline; gap:10px; margin-top:6px;">
        <span style="font-family:'IBM Plex Mono',monospace; font-size:28px; font-weight:500;
          color:#c8dff0;">{prediction:.4f}</span>
        <span style="font-family:'IBM Plex Mono',monospace; font-size:12px;
          color:{level_color};">{level}</span>
      </div>
      <div style="height:4px; background:#1a3a6a; border-radius:2px; margin-top:10px; overflow:hidden;">
        <div style="height:100%; width:{min(100, prediction*2000):.0f}%;
          background:{level_color}; border-radius:2px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_events:
    if "event" in news.columns:
        event_news = news[news["event"] == 1][["title", "sentiment"]].head(15)
        st.markdown("**⚠ Detected Geopolitical Events**")
        st.dataframe(event_news, use_container_width=True, height=260)
    else:
        st.info("Run the news pipeline to detect events.")



# -----------------------------------------------------------------------
# 3D INTERACTIVE GLOBE
# -----------------------------------------------------------------------

st.markdown('<div class="section-label">Geopolitical Risk Globe</div>', unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_world_geojson():
    import urllib.request, json
    url = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json"
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode())

try:
    world_topo = get_world_geojson()
    world_json = __import__('json').dumps(world_topo)
    globe_load_error = False
except Exception:
    world_json = "null"
    globe_load_error = True

GLOBE_HTML = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
  body {{ margin:0; background:#05080f; }}
  #gc {{ display:block; width:100%; cursor:grab; }}
  #gc:active {{ cursor:grabbing; }}
  #tt {{
    position:fixed; pointer-events:none; display:none;
    background:rgba(5,8,15,0.95); border:0.5px solid #1a3a6a;
    border-radius:6px; padding:8px 12px;
    font-family:'IBM Plex Mono',monospace; font-size:11px;
    color:#e0e8f0; z-index:999; max-width:210px; line-height:1.7;
  }}
  .cb {{
    font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:0.05em;
    padding:4px 10px; border-radius:4px; cursor:pointer;
    border:0.5px solid #1a3a6a; background:#0d1a2e; color:#7aabdf;
  }}
  .cb.active {{ background:#1a3a6a; color:#e0f0ff; }}
</style>
<div style="background:#05080f; padding:8px 0 4px;">
  <div style="display:flex;align-items:center;justify-content:space-between;padding:0 4px 8px;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#4a6a8a;">DRAG TO ROTATE &nbsp;·&nbsp; SCROLL TO ZOOM &nbsp;·&nbsp; HOVER FOR DETAILS</span>
    <div style="display:flex;gap:6px;">
      <button class="cb active" id="br" onclick="setMode('risk')">Risk</button>
      <button class="cb" id="bs" onclick="setMode('sentiment')">Sentiment</button>
      <button class="cb" id="bp" onclick="setMode('supply')">Supply</button>
    </div>
  </div>
  <canvas id="gc"></canvas>
  <div id="tt"></div>
  <div style="display:flex;gap:16px;padding:8px 4px 0;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#7a9ab0;">
    <span><span style="width:8px;height:8px;border-radius:50%;background:#D85A30;display:inline-block;margin-right:4px;"></span>High risk</span>
    <span><span style="width:8px;height:8px;border-radius:50%;background:#EF9F27;display:inline-block;margin-right:4px;"></span>Medium risk</span>
    <span><span style="width:8px;height:8px;border-radius:50%;background:#1D9E75;display:inline-block;margin-right:4px;"></span>Low risk</span>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/d3-geo@3/dist/d3-geo.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script>
const TOPO = {world_json};
const HS=[
  {{name:"Iran",lat:32,lon:53,risk:0.88,sentiment:-0.82,supply:"2.1M bbl/d",event:"Strait of Hormuz threat"}},
  {{name:"Russia",lat:61.5,lon:105.3,risk:0.76,sentiment:-0.61,supply:"4.5M bbl/d",event:"EU sanctions tightened"}},
  {{name:"Ukraine",lat:48.4,lon:31.2,risk:0.71,sentiment:-0.55,supply:"Transit disrupted",event:"Pipeline conflict zone"}},
  {{name:"Iraq",lat:33.2,lon:44.3,risk:0.65,sentiment:-0.44,supply:"200k bbl disrupted",event:"Basra pipeline attack"}},
  {{name:"Israel",lat:31,lon:35,risk:0.60,sentiment:-0.38,supply:"Regional risk",event:"Escalation monitoring"}},
  {{name:"Libya",lat:26.3,lon:17.2,risk:0.69,sentiment:-0.51,supply:"1.2M bbl at risk",event:"Port blockade ongoing"}},
  {{name:"Nigeria",lat:9,lon:8.6,risk:0.58,sentiment:-0.35,supply:"Delta region risk",event:"Pipeline sabotage"}},
  {{name:"Venezuela",lat:6.4,lon:-66.6,risk:0.55,sentiment:-0.30,supply:"Output declining",event:"Sanction uncertainty"}},
  {{name:"Saudi Arabia",lat:23.8,lon:45,risk:0.32,sentiment:0.55,supply:"Stable 9.8M bbl/d",event:"OPEC+ output pledge"}},
  {{name:"China",lat:35.8,lon:104.1,risk:0.41,sentiment:0.02,supply:"Imports flat",event:"Demand uncertain"}},
  {{name:"United States",lat:37.1,lon:-95.7,risk:0.28,sentiment:0.48,supply:"SPR replenishment",event:"Strategic reserve buy"}},
];
let mode='risk';
function dc(h,m){{
  if(m==='sentiment') return h.sentiment<-0.3?'#D85A30':h.sentiment>0.2?'#1D9E75':'#EF9F27';
  if(m==='supply') return '#378ADD';
  return h.risk>0.7?'#D85A30':h.risk>0.5?'#EF9F27':'#1D9E75';
}}
function dr(h,m){{
  const b=m==='risk'?h.risk:m==='sentiment'?Math.abs(h.sentiment):0.5;
  return 5+b*10;
}}
function setMode(m){{
  mode=m;
  ['r','s','p'].forEach(k=>document.getElementById('b'+k).classList.remove('active'));
  document.getElementById('b'+m[0]).classList.add('active');
  draw();
}}
const canvas=document.getElementById('gc');
let W=canvas.parentElement.offsetWidth||700, H=Math.round(W*0.54);
canvas.width=W*devicePixelRatio; canvas.height=H*devicePixelRatio;
canvas.style.width=W+'px'; canvas.style.height=H+'px';
const ctx=canvas.getContext('2d');
ctx.scale(devicePixelRatio,devicePixelRatio);
let rot=[15,-20],drag=false,last=null,vel=[0,0],sc=1,pT=0;

const proj=d3Geo.geoOrthographic().scale(1).translate([0,0]).clipAngle(90);
const pg=d3Geo.geoPath(proj,ctx);

let geo=null;
if(TOPO){{
  geo={{
    land: topojson.feature(TOPO, TOPO.objects.land),
    countries: topojson.feature(TOPO, TOPO.objects.countries)
  }};
}}

function R(){{return(Math.min(W,H)/2-8)*sc;}}

function draw(){{
  const r=R(),cx=W/2,cy=H/2;
  proj.scale(r).translate([cx,cy]).rotate(rot);
  ctx.clearRect(0,0,W,H);

  const g=ctx.createRadialGradient(cx,cy,r*0.3,cx,cy,r*1.4);
  g.addColorStop(0,'#0a1628'); g.addColorStop(1,'#05080f');
  ctx.fillStyle=g; ctx.fillRect(0,0,W,H);

  const a=ctx.createRadialGradient(cx,cy,r*0.97,cx,cy,r*1.12);
  a.addColorStop(0,'rgba(30,80,160,0.22)'); a.addColorStop(1,'rgba(0,0,0,0)');
  ctx.beginPath(); ctx.arc(cx,cy,r*1.12,0,Math.PI*2); ctx.fillStyle=a; ctx.fill();

  ctx.beginPath(); pg({{type:'Sphere'}}); ctx.fillStyle='#0d1e35'; ctx.fill();

  const gr=d3Geo.geoGraticule()();
  ctx.beginPath(); pg(gr); ctx.strokeStyle='rgba(30,70,120,0.18)'; ctx.lineWidth=0.4; ctx.stroke();

  if(geo){{
    ctx.beginPath(); pg(geo.land); ctx.fillStyle='#142236'; ctx.fill();
    ctx.beginPath(); pg(geo.countries); ctx.strokeStyle='rgba(40,100,160,0.4)'; ctx.lineWidth=0.55; ctx.stroke();
    ctx.beginPath(); pg(geo.land); ctx.strokeStyle='rgba(50,120,200,0.25)'; ctx.lineWidth=0.8; ctx.stroke();
  }}

  ctx.beginPath(); pg({{type:'Sphere'}}); ctx.strokeStyle='rgba(40,100,200,0.35)'; ctx.lineWidth=1; ctx.stroke();

  const pulse=0.5+0.5*Math.sin(pT);
  HS.forEach(h=>{{
    const vis=d3Geo.geoDistance([h.lon,h.lat],[-rot[0],-rot[1]])<Math.PI/2;
    if(!vis)return;
    const [px,py]=proj([h.lon,h.lat]);
    const dotR=dr(h,mode),col=dc(h,mode);
    ctx.beginPath(); ctx.arc(px,py,dotR*(1.5+pulse*0.6),0,Math.PI*2); ctx.fillStyle=col+'18'; ctx.fill();
    ctx.beginPath(); ctx.arc(px,py,dotR*1.2,0,Math.PI*2); ctx.fillStyle=col+'30'; ctx.fill();
    ctx.beginPath(); ctx.arc(px,py,dotR,0,Math.PI*2); ctx.fillStyle=col+'bb'; ctx.fill();
    ctx.strokeStyle=col; ctx.lineWidth=1; ctx.stroke();
    ctx.font="500 9px 'IBM Plex Mono',monospace"; ctx.fillStyle='#c8dff0';
    ctx.textAlign='left'; ctx.fillText(h.name,px+dotR+3,py+3);
  }});
}}

function animate(){{
  pT+=0.04;
  if(!drag){{vel[0]*=0.92;vel[1]*=0.92;rot[0]+=vel[0]+0.12;rot[1]=Math.max(-80,Math.min(80,rot[1]+vel[1]));}}
  draw();
  requestAnimationFrame(animate);
}}

canvas.addEventListener('mousedown',e=>{{drag=true;last=[e.clientX,e.clientY];}});
window.addEventListener('mouseup',()=>{{drag=false;}});
window.addEventListener('mousemove',e=>{{
  if(drag&&last){{
    const dx=e.clientX-last[0],dy=e.clientY-last[1];
    vel=[dx*0.3,-dy*0.3];
    rot[0]+=dx*0.3;rot[1]=Math.max(-80,Math.min(80,rot[1]-dy*0.3));
    last=[e.clientX,e.clientY];
  }}
  const rect=canvas.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;
  const tt=document.getElementById('tt');
  let hit=null;
  HS.forEach(h=>{{
    const vis=d3Geo.geoDistance([h.lon,h.lat],[-rot[0],-rot[1]])<Math.PI/2;
    if(!vis)return;
    const[px,py]=proj([h.lon,h.lat]);
    if(Math.hypot(mx-px,my-py)<dr(h,mode)+4)hit=h;
  }});
  if(hit){{
    const v=mode==='supply'?hit.supply:mode==='sentiment'?(hit.sentiment>=0?'+':'')+hit.sentiment.toFixed(2):hit.risk.toFixed(2);
    const lbl=mode==='risk'?'Risk':mode==='sentiment'?'Sentiment':'Supply';
    tt.style.display='block';
    tt.style.left=(e.clientX+14)+'px';
    tt.style.top=(e.clientY-10)+'px';
    tt.innerHTML=`<b style="font-size:12px;color:#e8f4ff;">${{hit.name}}</b><br>${{lbl}}: <b>${{v}}</b><br><span style="color:#5a8aaa;">${{hit.event}}</span>`;
    canvas.style.cursor='pointer';
  }}else{{tt.style.display='none';canvas.style.cursor=drag?'grabbing':'grab';}}
}});
canvas.addEventListener('wheel',e=>{{
  e.preventDefault();sc=Math.max(0.6,Math.min(2.5,sc-e.deltaY*0.001));draw();
}},{{passive:false}});

animate();
</script>
"""

components.html(GLOBE_HTML, height=520, scrolling=False)