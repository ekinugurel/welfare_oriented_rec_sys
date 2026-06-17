"""Map-visualisation helpers for the Streamlit / deck.gl frontend.

Turns a completed simulation day (OSM mode) into deck.gl-ready structures:

* ``trip_paths`` — one record per trip with the route polyline and per-vertex
  timestamps, for a deck.gl ``TripsLayer`` (animated route trails along real
  streets — the SUMO/MATSim look).
* ``frames`` — for each time bin, every agent's position + state, drawn as
  coloured markers (and emoji icons) via ``ScatterplotLayer`` / ``TextLayer``:
  in transit (🚗/🚶/🚲/🚌) vs stationary at home (🏠), work (🏢) or leisure (🎉).

These are pure functions (no Streamlit dependency) so they can be unit-tested
and reused by a future web backend. They require OSM mode, where agent/trip
locations are ``(lat, lon)`` and routes follow the real network.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from geo import haversine_km

LatLon = Tuple[float, float]

DAY_MINUTES = 1440

# Emoji icons (TextLayer) and RGB colours (ScatterplotLayer / TripsLayer).
MODE_EMOJI = {"car": "🚗", "walk": "🚶", "bike": "🚲", "transit": "🚌"}
STATE_EMOJI = {"home": "🏠", "work": "🏢", "leisure": "🎉", "in_transit": "🚗"}
MODE_COLOR = {
    "car": [230, 80, 60],
    "walk": [80, 170, 90],
    "bike": [240, 180, 40],
    "transit": [70, 130, 220],
}
STATE_COLOR = {
    "home": [80, 170, 90],
    "work": [240, 150, 40],
    "leisure": [160, 90, 210],
    "in_transit": [230, 80, 60],
}


def _cumulative_km(path: List[LatLon]) -> List[float]:
    """Cumulative great-circle distance (km) along a polyline."""
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + haversine_km(path[i - 1], path[i]))
    return cum


def _position_at(path: List[LatLon], cum: List[float], frac: float) -> LatLon:
    """Interpolate a position along ``path`` at distance fraction ``frac``."""
    if len(path) == 1:
        return path[0]
    target = max(0.0, min(1.0, frac)) * cum[-1]
    for i in range(1, len(path)):
        if cum[i] >= target:
            seg = cum[i] - cum[i - 1]
            f = 0.0 if seg <= 0 else (target - cum[i - 1]) / seg
            lat = path[i - 1][0] + (path[i][0] - path[i - 1][0]) * f
            lon = path[i - 1][1] + (path[i][1] - path[i - 1][1]) * f
            return (lat, lon)
    return path[-1]


def build_timelines(sim) -> Dict[int, dict]:
    """Per-agent timeline of moving (trip) and stationary (activity) segments.

    Reconstructs each agent's day from its realised trips: between trips the
    agent waits at the previous destination, and the state there is the purpose
    of the trip that brought it (home/work/leisure).
    """
    net = sim.road_network
    timelines: Dict[int, dict] = {}
    for agent in sim.agents:
        trips = sorted(agent.trips, key=lambda t: t.depart_time)
        moves: List[dict] = []
        stays: List[Tuple[int, int, LatLon, str]] = []
        prev_end = 0
        prev_loc: LatLon = agent.home
        prev_state = "home"
        for tr in trips:
            if tr.depart_time > prev_end:
                stays.append((prev_end, tr.depart_time, prev_loc, prev_state))
            path = net.route_geometry_latlon(tr.origin, tr.destination)
            moves.append(
                {
                    "t0": tr.depart_time,
                    "t1": tr.arrival_time,
                    "path": path,
                    "cum": _cumulative_km(path),
                    "mode": tr.mode,
                    "purpose": tr.purpose,
                }
            )
            prev_end = tr.arrival_time
            prev_loc = tr.destination
            prev_state = tr.purpose
        stays.append((prev_end, DAY_MINUTES, prev_loc, prev_state))
        timelines[agent.id] = {"moves": moves, "stays": stays}
    return timelines


MAX_PATH_POINTS = 12  # downsample route polylines to bound the browser payload


def _downsample(path: List[LatLon], k: int) -> List[LatLon]:
    """Keep ~k evenly-spaced points (including both endpoints) from a polyline."""
    n = len(path)
    if n <= k:
        return path
    step = (n - 1) / (k - 1)
    return [path[round(i * step)] for i in range(k)]


def build_trip_paths(timelines: Dict[int, dict]) -> List[dict]:
    """Per-trip records for the moving-dot animation: a *downsampled* route polyline
    (≤MAX_PATH_POINTS points) with per-vertex timestamps (minutes). The dots
    interpolate smoothly along these; full street trails are not shipped, which is
    what keeps the payload small. (Full geometry can be fetched on demand later.)"""
    out: List[dict] = []
    for agent_id, tl in timelines.items():
        for mv in tl["moves"]:
            path = _downsample(mv["path"], MAX_PATH_POINTS)
            cum = _cumulative_km(path)
            total = cum[-1] or 1.0
            t0, t1 = mv["t0"], mv["t1"]
            timestamps = [round(t0 + (t1 - t0) * (c / total), 1) for c in cum]
            out.append(
                {
                    "agent_id": agent_id,
                    "path": [[round(lon, 5), round(lat, 5)] for (lat, lon) in path],  # deck wants [lon, lat]
                    "timestamps": timestamps,
                    "mode": mv["mode"],
                    "color": MODE_COLOR.get(mv["mode"], [200, 200, 200]),
                }
            )
    return out


def _marker_at(tl: dict, t: int):
    """Return (lat, lon, state, mode) for one agent at time t, or None."""
    for mv in tl["moves"]:
        if mv["t0"] <= t <= mv["t1"]:
            span = max(1, mv["t1"] - mv["t0"])
            pos = _position_at(mv["path"], mv["cum"], (t - mv["t0"]) / span)
            return pos[0], pos[1], "in_transit", mv["mode"]
    for (s0, s1, loc, state) in tl["stays"]:
        if s0 <= t <= s1:
            return loc[0], loc[1], state, None
    return None


def markers_at(timelines: Dict[int, dict], t: float) -> List[dict]:
    """Every agent's marker (position, state, colour) at an exact time t (minutes).

    In-transit agents are interpolated *along their real route polyline* (not
    snapped to an intersection); stationary agents sit at their activity location.
    Computing this on the fly at any t gives continuous, MATSim-style movement.
    """
    markers: List[dict] = []
    for agent_id, tl in timelines.items():
        m = _marker_at(tl, t)
        if m is None:
            continue
        lat, lon, state, mode = m
        label = f"agent {agent_id} · {state}" + (f" · {mode}" if mode else "")
        markers.append(
            {
                "agent_id": agent_id,
                "lat": lat,
                "lon": lon,
                "state": state,
                "mode": mode or "",
                "color": STATE_COLOR.get(state, [150, 150, 150]),
                "label": label,
            }
        )
    return markers


def day_layers(sim, offset: int = 0):
    """``(trip_paths, stays)`` for the sim's current day, timestamps shifted by
    ``offset`` minutes so successive days can be laid end-to-end (day d → offset
    d·1440) for a multi-day animation."""
    timelines = build_timelines(sim)
    trips = build_trip_paths(timelines)
    if offset:
        for tr in trips:
            tr["timestamps"] = [t + offset for t in tr["timestamps"]]
    stays = [
        {"agent_id": aid, "t0": s0 + offset, "t1": s1 + offset, "lat": loc[0], "lon": loc[1],
         "state": state, "color": STATE_COLOR.get(state, [150, 150, 150])}
        for aid, tl in timelines.items()
        for (s0, s1, loc, state) in tl["stays"]
    ]
    return trips, stays


def build_visualization(sim) -> dict:
    """Build the deck.gl payload: agent timelines, trip paths, POIs, view bounds.

    Agent markers are computed on the fly at any time via ``markers_at`` (so the
    payload is independent of playback speed); ``trip_paths`` feed a TripsLayer.
    """
    timelines = build_timelines(sim)
    net = sim.road_network
    south, west, north, east = net.bounds
    pois = [
        {
            "lon": p.location[1],
            "lat": p.location[0],
            "category": p.category,
            "label": f"{p.name} · {p.category}",
        }
        for p in sim.place_catalog
    ]
    # Stationary segments (agent parked at an activity) — the client interpolates
    # moving agents from trip_paths and draws parked ones from these.
    stays = [
        {"agent_id": agent_id, "t0": s0, "t1": s1, "lat": loc[0], "lon": loc[1],
         "state": state, "color": STATE_COLOR.get(state, [150, 150, 150])}
        for agent_id, tl in timelines.items()
        for (s0, s1, loc, state) in tl["stays"]
    ]
    return {
        "timelines": timelines,
        "trip_paths": build_trip_paths(timelines),
        "stays": stays,
        "pois": pois,
        "view": {"latitude": net.center[0], "longitude": net.center[1]},
        "bounds": {"south": south, "west": west, "north": north, "east": east},
    }
