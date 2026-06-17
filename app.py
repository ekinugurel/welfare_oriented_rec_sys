"""Streamlit frontend for the welfare-oriented activity-travel ABM on a real
OpenStreetMap street network (New York by default).

Run with:
    streamlit run app.py

Left panel controls the simulation (city, population, days, seed, recommender
treatment, modes). The map animates the chosen day — agents move along real
streets (TripsLayer) with markers for in-car / walking / at-home / at-work /
at-leisure states. The lower panels graph the run's outcomes.

Heavy logic lives in importable, Streamlit-free functions (``build_network``,
``run_simulation``, ``make_deck``) so it can be tested and reused by a future
web backend; ``main()`` holds the UI.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

import params
import viz
from geo import build_road_network, haversine_km
from simulation import Simulation

TREATMENTS = ["No RS", "Standard RS", "PUP", "RM", "PUP+RM"]
_WELFARE_MODE = {"PUP": "pup", "RM": "rm", "PUP+RM": "pup_rm"}

# (city-preset key, friendly label) — shown as area options on the main screen.
AREAS = [
    ("nyc_manhattan", "Manhattan"),
    ("nyc_lower_manhattan", "Lower Manhattan"),
    ("brooklyn_full", "Brooklyn (full)"),
    ("brooklyn_downtown_park_slope", "Downtown BK / Park Slope"),
    ("brooklyn_south_prospect_bay_ridge", "South BK · Prospect / Bay Ridge"),
]


# ── Engine (Streamlit-free, testable) ────────────────────────────────────────

def build_network(city: str):
    """Build/load the OSM network for a city preset (see params.GEO_PARAMS)."""
    return build_road_network(city)


def _make_recommender_factory(treatment: str, pup_alpha: float, rm_epsilon: float):
    """Return a recommender_factory wiring the welfare gate, or None for the
    pass-through Standard RS / No RS conditions."""
    if treatment not in _WELFARE_MODE:
        return None
    from welfare_layer import TravelCostEstimator, WelfareAwareOrchestrator

    mode = _WELFARE_MODE[treatment]

    def factory(_sim, base_stack):
        return WelfareAwareOrchestrator(
            base_stack,
            TravelCostEstimator(),
            mode=mode,
            pup_alpha=pup_alpha,
            rm_epsilon=rm_epsilon,
            coord_scale_km=1.0,
            coord_distance_km=haversine_km,
        )

    return factory


def run_simulation(
    network,
    *,
    num_agents: int,
    num_days: int,
    seed: int,
    treatment: str,
    multimodal: bool,
    pup_alpha: float,
    rm_epsilon: float,
    poi_csv_path: Optional[str] = None,
    progress=None,
):
    """Run the ABM on a real network and build the visualization payload.

    ``progress(day, total)`` is called at the start of each simulated day.
    Returns ``(visualization, day_summaries, final_summary)``.
    """
    # Mode model: paper default is car-only; the frontend can re-enable the
    # multimodal model (car + walk + bike) on the same network. Transit/subway is
    # excluded per design — all modes route on the drive network.
    params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"] = not multimodal

    sim = Simulation(
        num_agents=num_agents,
        seed=seed,
        use_recommenders=(treatment != "No RS"),
        persona_csv_path=params.NYC_PERSONA_CSV_PATH,
        road_network=network,
        poi_csv_path=poi_csv_path,  # real NYC POIs when given, else synthetic
        disabled_modes=("transit",),
        recommender_factory=_make_recommender_factory(treatment, pup_alpha, rm_epsilon),
    )
    # Capture each day's trajectories and lay them end-to-end so the animation
    # spans all simulated days (the sim resets agent.trips between days).
    day_trips, day_stays = [], []

    def _capture(day_idx, s):
        trips, stays = viz.day_layers(s, offset=day_idx * 1440)
        day_trips.extend(trips)
        day_stays.extend(stays)

    day_summaries = sim.run_days(num_days, progress=progress, on_day_complete=_capture)
    net = sim.road_network
    visualization = {
        "trip_paths": day_trips,
        "stays": day_stays,
        "pois": [{"lon": p.location[1], "lat": p.location[0], "category": p.category,
                  "label": f"{p.name} · {p.category}"} for p in sim.place_catalog],
        "view": {"latitude": net.center[0], "longitude": net.center[1]},
        "num_days": num_days,
    }
    final = sim.summarize()
    # The paper-relevant welfare signal is the net utility of leisure trips (the
    # RS-influenced ones); avg_trip_utility is dominated by mandatory commutes.
    leisure_utils = [t.utility for a in sim.agents for t in a.trips if t.purpose == "leisure"]
    final["mean_leisure_net_utility"] = sum(leisure_utils) / len(leisure_utils) if leisure_utils else 0.0
    final["poi_count"] = len(sim.place_catalog)
    return visualization, day_summaries, final


def make_deck(
    visualization: dict,
    current_time: float,
    *,
    show_trips: bool = True,
    show_pois: bool = True,
    trail_length: int = 60,
    zoom: float = 11.5,
) -> pdk.Deck:
    """Build the pydeck Deck at an exact animation time (minutes since midnight).

    Agent positions are computed continuously at ``current_time`` (interpolated
    along real routes), so playback moves smoothly rather than snapping to bins.
    """
    markers = viz.markers_at(visualization["timelines"], current_time)
    layers = []

    if show_pois and visualization.get("pois"):
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=visualization["pois"],
                get_position="[lon, lat]",
                get_fill_color=[110, 110, 120],
                get_radius=6,
                radius_min_pixels=1,
                radius_max_pixels=3,
                opacity=0.35,
                pickable=True,
            )
        )
    if show_trips:
        layers.append(
            pdk.Layer(
                "TripsLayer",
                data=visualization["trip_paths"],
                get_path="path",
                get_timestamps="timestamps",
                get_color="color",
                current_time=current_time,
                trail_length=trail_length,
                width_min_pixels=2,
                rounded=True,
                opacity=0.6,
            )
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=markers,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=14,
            radius_min_pixels=2,
            radius_max_pixels=5,
            pickable=True,
            opacity=0.9,
        )
    )

    view = pdk.ViewState(
        latitude=visualization["view"]["latitude"],
        longitude=visualization["view"]["longitude"],
        zoom=zoom,
        pitch=0,
    )
    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_provider="carto",
        map_style="light",
        tooltip={
            "html": "<b>{label}</b>",
            "style": {"backgroundColor": "#1f2430", "color": "#fff",
                      "fontSize": "12px", "padding": "4px 8px"},
        },
    )


def _hhmm(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


_TRIPS_HTML = """<!doctype html><html><head><meta charset="utf-8"/>
<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet"/>
<script src="https://unpkg.com/deck.gl@8.9.36/dist.min.js"></script>
<style>
 html,body{margin:0}#wrap{position:relative;height:600px;font-family:system-ui,sans-serif}
 #map{position:absolute;inset:0}
 #ctl{position:absolute;left:10px;right:10px;bottom:10px;z-index:2;display:flex;gap:10px;align-items:center;
      background:rgba(31,36,48,.85);color:#fff;padding:8px 12px;border-radius:8px}
 #ctl button{background:#3b82f6;color:#fff;border:0;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:14px}
 #ctl input[type=range]{flex:1}#clock{font-variant-numeric:tabular-nums;min-width:90px;text-align:center;white-space:nowrap}
</style></head><body><div id="wrap"><div id="map"></div>
<div id="ctl"><button id="pp">&#9208; Pause</button><span id="clock">00:00</span>
<input id="scrub" type="range" min="0" max="1440" step="1" value="0"/>
<span style="white-space:nowrap">&#9193; <span id="spdval"></span></span>
<input id="spd" type="range" min="30" max="1440" step="30" style="width:100px;flex:none"/></div></div>
<script>
const D=__PAYLOAD__;
const fmt=m=>String(Math.floor(m/60)%24).padStart(2,'0')+':'+String(Math.floor(m)%60).padStart(2,'0');
const TMAX=D.days*1440,clk=t=>'Day '+(Math.floor(t/1440)+1)+' · '+fmt(t%1440);
document.getElementById('scrub').max=TMAX;
const tByA={},sByA={},ids=new Set();
D.trips.forEach(t=>{ids.add(t.agent_id);(tByA[t.agent_id]=tByA[t.agent_id]||[]).push(t);});
D.stays.forEach(s=>{ids.add(s.agent_id);(sByA[s.agent_id]=sByA[s.agent_id]||[]).push(s);});
const AG=[...ids];
function interp(tr,t){const ts=tr.timestamps,p=tr.path;if(t<=ts[0])return p[0];if(t>=ts[ts.length-1])return p[p.length-1];
 for(let i=1;i<ts.length;i++){if(ts[i]>=t){const f=(t-ts[i-1])/((ts[i]-ts[i-1])||1);
 return [p[i-1][0]+(p[i][0]-p[i-1][0])*f,p[i-1][1]+(p[i][1]-p[i-1][1])*f];}}return p[p.length-1];}
function markers(t){const o=[];for(const a of AG){let d=false;for(const tr of (tByA[a]||[])){
 if(t>=tr.timestamps[0]&&t<=tr.timestamps[tr.timestamps.length-1]){o.push({position:interp(tr,t),color:[230,80,60],label:'Agent '+a+' &middot; in transit'+(tr.mode?' ('+tr.mode+')':'')});d=true;break;}}
 if(!d)for(const s of (sByA[a]||[])){if(t>=s.t0&&t<=s.t1){o.push({position:[s.lon,s.lat],color:s.color,label:'Agent '+a+' &middot; '+(s.state||'')});break;}}}return o;}
const map=new maplibregl.Map({container:'map',style:'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
 center:[D.view.longitude,D.view.latitude],zoom:D.zoom,attributionControl:false});
const overlay=new deck.MapboxOverlay({interleaved:false,layers:[],pickingRadius:5,
 getTooltip:({object})=>object&&object.label?{html:'<b>'+object.label+'</b>',
 style:{background:'#1f2430',color:'#fff',fontSize:'12px',padding:'5px 9px',borderRadius:'6px'}}:null});
let time=0,playing=true,speed=D.speed,last=performance.now();
function render(){overlay.setProps({layers:[
 new deck.ScatterplotLayer({id:'pois',data:D.pois,getPosition:d=>[d.lon,d.lat],getFillColor:[110,110,120],getRadius:4,radiusMinPixels:1,radiusMaxPixels:2.5,opacity:.4,pickable:true,autoHighlight:true,highlightColor:[80,140,255,200]}),
 new deck.ScatterplotLayer({id:'ag',data:markers(time),getPosition:d=>d.position,getFillColor:d=>d.color,getRadius:14,radiusMinPixels:3,radiusMaxPixels:6,opacity:.95,pickable:true,autoHighlight:true,highlightColor:[255,255,255,180]})
]});document.getElementById('clock').textContent=clk(time);document.getElementById('scrub').value=time;}
function loop(now){const dt=(now-last)/1000;last=now;if(playing){time+=speed*dt;if(time>TMAX)time=0;}render();requestAnimationFrame(loop);}
map.on('load',()=>{map.addControl(overlay);requestAnimationFrame(loop);});
document.getElementById('pp').onclick=function(){playing=!playing;this.innerHTML=playing?'&#9208; Pause':'&#9654; Play';};
document.getElementById('scrub').oninput=function(){time=+this.value;playing=false;document.getElementById('pp').innerHTML='&#9654; Play';render();};
const _spd=document.getElementById('spd'),_sv=document.getElementById('spdval');_spd.value=speed;
function _showSpd(){const s=1440/speed;_sv.textContent=(s<1?s.toFixed(1):Math.round(s))+'s/day';}_showSpd();
_spd.oninput=function(){speed=+this.value;_showSpd();};
</script></body></html>"""


def build_trips_html(visualization: dict, *, trail_min: int = 60, speed: int = 240, zoom: float = 11.5) -> str:
    """Self-contained deck.gl + MapLibre page that animates the day client-side.

    Renders once and runs its own requestAnimationFrame loop (Play/Pause/scrub
    inside), so playback is smooth and never re-runs Python — eliminating the
    per-timestep flicker of the Streamlit-rerun approach.
    """
    payload = {
        "trips": visualization["trip_paths"],
        "stays": visualization["stays"],
        "pois": [{"lon": p["lon"], "lat": p["lat"], "label": p["label"]} for p in visualization["pois"]],
        "view": visualization["view"],
        "zoom": zoom,
        "trail": trail_min,
        "speed": speed,  # simulated minutes per real second
        "days": visualization.get("num_days", 1),
    }
    return _TRIPS_HTML.replace("__PAYLOAD__", json.dumps(payload))


def _resolve_poi_path() -> Optional[str]:
    """Locate the filtered NYC POI file, robust to a remapped HOME env var.

    Tries the configured path first, then the user's real home directory (from the
    password database, not $HOME) so the real POIs load even when the launching
    process has a non-standard environment.
    """
    candidates = [params.NYC_POI_CSV_PATH]
    try:
        import pwd

        real_home = pwd.getpwuid(os.getuid()).pw_dir
        candidates.append(os.path.join(real_home, ".cache", "welfare_rs", "pois", "nyc_leisure_pois.csv"))
    except Exception:
        pass
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


# ── Streamlit-cached wrappers ────────────────────────────────────────────────

_get_network = st.cache_resource(show_spinner="Loading street network…")(build_network)


def _run_simulation(city, num_agents, num_days, seed, treatment, multimodal,
                    pup_alpha, rm_epsilon, use_real_pois, progress=None):
    """Run (or fetch a cached) simulation, keyed by config in ``session_state``.

    We cache here rather than with ``@st.cache_data`` because the live per-day
    ``progress`` callback renders Streamlit elements; cache_data records and
    replays those on cache hits, which fails for elements created outside the
    function (CacheReplayClosureError, e.g. when pressing Play). With a plain
    session_state cache, ``progress`` fires only on an actual run and hits return
    instantly with no Streamlit calls. ``progress(0, total)`` signals the build
    ("preparing") phase; ``progress(day, total)`` fires per simulated day.
    """
    key = (city, int(num_agents), int(num_days), int(seed), treatment, bool(multimodal),
           round(float(pup_alpha), 4), round(float(rm_epsilon), 4), bool(use_real_pois))
    cache = st.session_state.setdefault("_sim_cache", {})
    if key in cache:
        return cache[key]
    if progress is not None:
        progress(0, int(num_days))
    poi_csv_path = _resolve_poi_path() if use_real_pois else None
    result = run_simulation(
        _get_network(city),
        num_agents=num_agents, num_days=num_days, seed=int(seed), treatment=treatment,
        multimodal=multimodal, pup_alpha=pup_alpha, rm_epsilon=rm_epsilon,
        poi_csv_path=poi_csv_path, progress=progress,
    )
    if len(cache) > 16:  # keep the session cache bounded
        cache.pop(next(iter(cache)))
    cache[key] = result
    return result


# ── UI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Welfare-RS · NYC", layout="wide")
    st.title("Welfare-oriented activity-travel simulation — New York")

    # ── Settings (sidebar) ───────────────────────────────────────────────────
    sb = st.sidebar
    sb.header("Settings")
    num_agents = int(sb.number_input(
        "Agents", min_value=1, max_value=5000, value=80, step=1,
        help="Up to 288 use distinct personas; beyond that, personas — and their home "
             "locations — repeat (work + random draws still differ). Regenerate the persona "
             "file with more rows for more distinct homes.",
    ))
    num_days = int(sb.number_input("Days", min_value=1, value=3, step=1))
    seed = sb.number_input("Random seed", value=42, step=1)
    treatment = sb.selectbox("Recommender", TREATMENTS, index=1)
    multimodal = sb.toggle("Multimodal (car + walk + bike)", value=False)
    pois_available = _resolve_poi_path() is not None
    use_real_pois = sb.toggle(
        "Real NYC POIs", value=pois_available, disabled=not pois_available,
        help="Use the filtered NYC POI dataset (run data/filter_nyc_pois.py first). "
             "Off → synthetic POIs placed on the network.",
    )

    pup_alpha, rm_epsilon = 0.6, 0.3
    if treatment in ("PUP", "PUP+RM"):
        pup_alpha = sb.slider("PUP α (min P[U≥0])", 0.0, 1.0, 0.6, 0.05)
    if treatment in ("RM", "PUP+RM"):
        rm_epsilon = sb.slider("RM ε (regret ceiling)", 0.0, 1.5, 0.3, 0.05)

    # ── Area selection (main screen) ─────────────────────────────────────────
    if "area" not in st.session_state:
        st.session_state.area = AREAS[0][0]
    st.subheader("Choose an area")
    for col, (key, label) in zip(st.columns(len(AREAS)), AREAS):
        if col.button(label, use_container_width=True,
                      type="primary" if st.session_state.area == key else "secondary"):
            st.session_state.area = key

    # The current widgets form a *pending* config. Nothing runs until Run is
    # pressed, which commits this snapshot; changing any setting/area afterwards
    # does NOT re-run — results stay on the last committed config.
    live_cfg = {
        "city": st.session_state.area, "num_agents": num_agents, "num_days": num_days,
        "seed": int(seed), "treatment": treatment, "multimodal": multimodal,
        "use_real_pois": use_real_pois, "pup_alpha": pup_alpha, "rm_epsilon": rm_epsilon,
    }
    if st.button("▶ Run simulation", type="primary"):
        st.session_state.committed = live_cfg

    cfg = st.session_state.get("committed")
    if cfg is None:
        st.info("Pick an area, set the options on the left, then press **▶ Run simulation**.")
        return
    if cfg != live_cfg:
        st.caption("⚙️ Settings changed — press **▶ Run simulation** to apply.")

    # ── Run the committed config ──────────────────────────────────────────────
    network = _get_network(cfg["city"])

    # Live per-day progress (only fills on a cache miss; cached re-runs are instant).
    status = st.empty()

    def _on_day(day, total):
        if day == 0:
            status.progress(0.0, text="Preparing simulation…")
        else:
            status.progress(day / total, text=f"Running day {day} of {total}…")

    visualization, day_summaries, summary = _run_simulation(
        cfg["city"],
        num_agents=cfg["num_agents"],
        num_days=cfg["num_days"],
        seed=cfg["seed"],
        treatment=cfg["treatment"],
        multimodal=cfg["multimodal"],
        pup_alpha=cfg["pup_alpha"],
        rm_epsilon=cfg["rm_epsilon"],
        use_real_pois=cfg["use_real_pois"],
        progress=_on_day,
    )
    status.empty()
    poi_src = "real NYC dataset" if (cfg["use_real_pois"] and pois_available) else "synthetic"
    sb.caption(f"Network: {network.num_base_nodes:,} road intersections")
    sb.caption(f"POIs: {summary.get('poi_count', 0):,} ({poi_src})")

    # ── Map + animation (smooth client-side deck.gl; starts at 00:00) ─────────
    st.subheader(f"Activity-travel over the day · {dict(AREAS)[cfg['city']]} · {cfg['treatment']}")
    components.html(
        build_trips_html(visualization, zoom=11.5),
        height=600,
    )
    st.caption(
        "**Dots = agents:** 🔴 in transit · 🟢 home · 🟠 work · 🟣 leisure  ·  "
        "**grey dots = POIs**  ·  ▶/⏸, scrub, and adjust speed on the map · hover a dot for details."
    )

    # ── Results ──────────────────────────────────────────────────────────────
    st.subheader("Results (final day)")
    leisure_trips = sum(summary.get("leisure_subtype_counts", {}).values())
    k = st.columns(5)
    k[0].metric("Trips", summary.get("trips", 0))
    k[1].metric("Leisure trips", leisure_trips)
    k[2].metric("Rec. acceptance", f"{summary.get('recommendation_acceptance_rate', 0):.0%}")
    k[3].metric("Leisure net utility", f"{summary.get('mean_leisure_net_utility', 0):.3f}",
                help="Mean net utility of leisure trips (activity benefit − travel cost). "
                     "The welfare-relevant signal: Standard RS often pushes this below No RS.")
    k[4].metric("Avg travel time", f"{summary.get('avg_travel_time_min', 0):.1f} min")

    c1, c2 = st.columns(2)
    mode_share = summary.get("mode_share", {})
    if mode_share:
        c1.markdown("**Mode share**")
        c1.bar_chart(pd.Series(mode_share, name="share"))
    subtypes = summary.get("leisure_subtype_counts", {})
    if subtypes:
        c2.markdown("**Leisure activity mix**")
        c2.bar_chart(pd.Series(subtypes, name="trips"))

    if len(day_summaries) > 1:
        st.markdown("**Daily trends**")
        df = pd.DataFrame(day_summaries).set_index("day")
        cols = [c for c in ["avg_trip_utility", "recommendation_acceptance_rate",
                            "feedback_like_rate", "avg_travel_time_min"] if c in df]
        st.line_chart(df[cols])


if __name__ == "__main__":
    main()
