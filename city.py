"""City spatial environment for the travel-behaviour ABM."""

from __future__ import annotations

import copy
import random

import numpy as np

import params


class City:
    """Spatial environment and global context.

    The city is a grid. Each cell has a zone type. The context includes
    policy and environmental conditions, plus social norms and practices.
    """

    def __init__(self, size_x, size_y, block_km=None, seed=42):
        cp = params.CITY_PARAMS
        self.size_x = size_x
        self.size_y = size_y
        self.block_km = block_km if block_km is not None else cp["block_km"]
        self.rng = random.Random(seed)

        # Context factors (C) from MTTC — deep-copy so runtime patches don't
        # mutate the template in params.py.
        self.context = copy.deepcopy(cp["default_context"])
        self.context["community_mode_bias"] = dict(cp["community_mode_bias"])

        self._build_zones()

        area = size_x * size_y
        self.road_capacity = max(
            cp["road_capacity_floor"],
            int(area * cp["road_capacity_multiplier"]),
        )
        self.transit_capacity = max(
            cp["transit_capacity_floor"],
            int(area * cp["transit_capacity_multiplier"]),
        )

    def _build_zones(self):
        """Assign a zone type to each grid cell using a simple probability mix."""
        probs = params.CITY_PARAMS["zone_probabilities"]
        choices = list(probs.keys())
        weights = [probs[k] for k in choices]
        zones = []
        for _ in range(self.size_x * self.size_y):
            zones.append(self.rng.choices(choices, weights=weights, k=1)[0])

        self.zones = np.array(zones, dtype=object).reshape(self.size_x, self.size_y)
        self.zone_index = {k: [] for k in choices}
        for x in range(self.size_x):
            for y in range(self.size_y):
                self.zone_index[self.zones[x, y]].append((x, y))

    def distance_km(self, a, b):
        """Return Manhattan distance in kilometers (grid-based)."""
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * self.block_km

    def sample_location(self, zone_type):
        """Sample a random location for a given zone type."""
        pool = self.zone_index.get(zone_type, [])
        if not pool:
            return (self.rng.randrange(self.size_x), self.rng.randrange(self.size_y))
        return self.rng.choice(pool)

    def sample_location_weighted(self, zone_weights):
        """Sample a location by weighted zone types."""
        zones = []
        weights = []
        for zone_type, weight in zone_weights.items():
            if weight <= 0:
                continue
            if self.zone_index.get(zone_type):
                zones.append(zone_type)
                weights.append(weight)
        if not zones:
            return self.sample_location("mixed")
        chosen_zone = self.rng.choices(zones, weights=weights, k=1)[0]
        return self.sample_location(chosen_zone)

    def community_mode_bias(self, mode):
        """Return a social-practice bias for a mode (SPT)."""
        return self.context.get("community_mode_bias", {}).get(mode, 0.0)
