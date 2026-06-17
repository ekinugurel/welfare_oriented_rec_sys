# welfare_oriented_rec_sys

Agent-based simulation for **"Towards welfare-oriented recommendations in
activity-travel behavior"** (RecSys '26), plus an OpenStreetMap road-network
backend and a Streamlit map frontend.

The original model runs on a synthetic grid. This repo adds an **opt-in OSM mode**
that replaces the grid with a real street network (New York by default), so the
travel cost `C_travel` — the quantity the paper's PUP/RM welfare gate hinges on —
is a real network route rather than Manhattan distance on a random grid. The grid
model is unchanged and remains the default when no network is supplied.

## What's new

| File | Role |
|------|------|
| `geo.py` | `RoadNetwork`: downloads/caches an OSM street network, snaps coordinates to nodes, and answers routing queries (distance, geometry, single-source). Pure geography, no ABM knowledge. |
| `city.py` / `simulation.py` | Take an optional `road_network`. In OSM mode, locations are `(lat, lon)`, distance is a network route, personas snap to real nodes, and POIs are placed on the network. |
| `recommender_systems.py` / `welfare_layer.py` | Injectable distance function — straight-line **haversine** in OSM mode (the paper's "as the crow flies" RS proximity), while the realised trip uses the network route. |
| `viz.py` | Turns a simulated day into deck.gl structures: animated route trails + per-time-bin agent markers (in-car / walking / at-home / at-work / at-leisure). |
| `app.py` | Streamlit frontend: controls, animated map, and result graphs. |
| `data/make_nyc_personas.py` | Generates NY personas from the Seattle file by sampling home locations on the network. |

## Setup

Use the `trex` conda env (Python 3.11). Dependencies are in `requirements.txt`
(`osmnx`, `geopandas`, `shapely`, `networkx`, `streamlit`, `pydeck`, …):

```bash
pip install -r requirements.txt
```

OSM usage is **not** metered: the street network is downloaded once via OSMnx and
cached to GraphML under `~/.cache/welfare_rs/` (override with `WELFARE_RS_CACHE`);
all routing is local. Only the map basemap tiles are fetched live (Carto, free).

## Run the frontend

```bash
streamlit run app.py
```

The left panel controls the city/area, population, days, seed, recommender
treatment (**No RS / Standard RS / PUP / RM / PUP+RM**), and modes (car-only vs.
car+walk+bike+transit). The map animates the day; the lower panels graph the
outcomes. Switching to PUP/RM shows the welfare gate abstaining and recovering
utility that Standard RS destroys.

## Regenerate NY personas (optional)

The repo ships `data/synthetic_personas_..._NewYork.csv`. To regenerate (e.g. for
a different area), pick a city preset from `params.GEO_PARAMS['cities']`:

```bash
python data/make_nyc_personas.py --city nyc_manhattan --out data/<name>.csv
```

## POI data

The simulation uses a real POI dataset (`poi_children_merged_by_wkt.csv`, ~375 MB,
all US, Placekey/NAICS schema). Filter it to NYC leisure POIs **once**:

```bash
python data/filter_nyc_pois.py --source ~/Desktop/poi_children_merged_by_wkt.csv
# -> ~/.cache/welfare_rs/pois/nyc_leisure_pois.csv  (params.NYC_POI_CSV_PATH)
```

This keeps ~64k NYC POIs whose NAICS code maps to a leisure category (mapping in
`params.POI_PARAMS`). `Simulation._load_osm_poi_catalog` then loads them, restricts
to the active network's bounds, and caps per category (`max_per_category`). The
source has no ratings/reviews, so those prominence signals are synthesised
deterministically per place. The frontend's **Real NYC POIs** toggle uses this file
when present (otherwise it falls back to synthetic POIs on the network).

POIs are **inserted into the road graph as mid-block nodes** (`RoadNetwork.
add_pois_as_nodes`): each POI splits its nearest edge at the projected point, so a
block becomes `intersection → poi → poi → intersection` and routing to/from a POI
is granular within the block (rather than snapping to the nearest intersection).
Personas/work/organic destinations still sample only real intersections.

## Notes / next steps

- Work locations are random network nodes (the simple choice) — this makes some
  commutes long; bound them to a radius of home for tighter trips if desired.
- Congestion is the original global BPR fed real network lengths; per-edge
  congestion from OSM road class/lanes is a natural next step.
- Transit/bike/walk reuse the drive-network route with mode-specific
  speed/cost/wait from `params.MODE_PARAMS` (no GTFS).
