"""Utility-based welfare layer for the recommender system stack.

Implements:
- Continuous feedback signal (extends binary like/dislike → satisfaction + travel cost)
- Travel cost estimation from feedback history
- PUP (Positive Utility Probability) gating
- RM (Regret Minimization) gating
- WelfareAwareOrchestrator: wraps LeisureRSOrchestrator with welfare constraints
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import params
from recommender_systems import (
    LeisureRSOrchestrator,
    Place,
    Recommendation,
    UserContext,
    euclidean_distance_km,
)


DEFAULT_SCORE_TO_UTILITY_MODE = "subtype_base_plus_score"
LEGACY_SCORE_TO_UTILITY_MODE = "identity"
DEFAULT_SCORE_UTILITY_SCALE = max(
    seg.get("activity_utility", 0.0)
    for seg in params.LEISURE_SEGMENTS.values()
)


# ── Continuous feedback ──────────────────────────────────────────────────────

@dataclass
class ContinuousFeedback:
    """Extended feedback signal beyond binary like/dislike.

    Attributes:
        user_id: Agent identifier.
        place_id: Place that was visited.
        source: Recommender source name.
        satisfaction: Continuous satisfaction score in [-1, 1].
        realized_travel_cost: Actual travel cost experienced (generalized cost).
        activity_utility: Realized activity utility.
        travel_utility: Realized travel utility (negative = costly).
        net_utility: satisfaction proxy = activity_utility + travel_utility.
    """

    user_id: int
    place_id: str
    source: str
    satisfaction: float  # [-1, 1]
    realized_travel_cost: float  # generalized cost in $-equivalent
    activity_utility: float
    travel_utility: float
    net_utility: float


def compute_continuous_feedback(
    trip,
    p_like: float,
    feedback_sensitivity: float = 1.0,
    agent_vot: Optional[float] = None,
    beta_time: Optional[float] = None,
    beta_cost: Optional[float] = None,
) -> ContinuousFeedback:
    """Convert a Trip + binary feedback signal into a continuous feedback record.

    The satisfaction score maps from the sigmoid probability p_like
    into a continuous [-1, 1] range, preserving the agent's nuanced
    evaluation rather than collapsing to binary.

    ``realized_travel_cost`` reports the generalized cost the agent
    *actually incurred* on this trip, using the agent's true value of
    time and time/cost sensitivities per Eq.~7 of the manuscript:

        C = τ · v_i · β_i^time  +  κ · β_i^cost

    Passing ``agent_vot``, ``beta_time``, and ``beta_cost`` is strongly
    preferred — that is the path that exercises the TCE's VOT-learning
    signal. When any of the three is ``None`` we fall back to
    ``params.WELFARE_LAYER_PARAMS["feedback_realized_vot"]`` (prior
    anchor) and β = 1. That fallback is kept for backward compatibility
    with callers that don't plumb agent-level preferences through, but
    it suppresses VOT learning because the observed cost then matches
    the TCE's prior exactly.

    Parameters
    ----------
    trip : Trip
        The completed trip with utility components.
    p_like : float
        The probability of liking (from agent's feedback evaluation).
    feedback_sensitivity : float
        Agent-specific sensitivity multiplier.
    agent_vot : float, optional
        The agent's true value of time ($/hr).  When provided, the
        realized generalized cost is computed using the agent's true
        preferences rather than a fixed prior anchor.
    beta_time, beta_cost : float, optional
        The agent's time/cost sensitivity weights.  Default to 1.0 when
        only a subset is supplied.

    Returns
    -------
    ContinuousFeedback
        Continuous feedback record.
    """
    # Map p_like ∈ [0, 1] → satisfaction ∈ [-1, 1]
    satisfaction = (2.0 * p_like - 1.0) * min(2.0, max(0.2, feedback_sensitivity))
    satisfaction = max(-1.0, min(1.0, satisfaction))

    # Generalized travel cost per Eq.~7 (manuscript) using the agent's
    # *true* value of time and sensitivity weights when provided, so
    # that the TCE can actually learn (the residual between what the
    # user paid and what the TCE currently predicts is what drives the
    # asymmetric VOT update).
    if agent_vot is None:
        agent_vot = params.WELFARE_LAYER_PARAMS["feedback_realized_vot"]
    if beta_time is None:
        beta_time = 1.0
    if beta_cost is None:
        beta_cost = 1.0

    realized_travel_cost = (
        trip.cost * beta_cost
        + (trip.travel_time_min / 60.0) * agent_vot * beta_time
    )

    net_utility = trip.activity_utility + trip.travel_utility

    return ContinuousFeedback(
        user_id=trip.agent_id,
        place_id=trip.recommended_place_id,
        source=trip.recommendation_source,
        satisfaction=satisfaction,
        realized_travel_cost=realized_travel_cost,
        activity_utility=trip.activity_utility,
        travel_utility=trip.travel_utility,
        net_utility=net_utility,
    )


# ── Travel cost estimation ───────────────────────────────────────────────────

class TravelCostEstimator:
    """Learns per-user travel cost parameters from feedback history.

    Maintains running estimates of:
    - β̂_i^time: user's marginal disutility of travel time ($/min)
    - β̂_i^cost: user's marginal disutility of monetary cost ($/$ spent)
    - v̂_i: user's revealed value of time ($/hr)

    Uses exponential moving average over realized trip costs.

    All tunable parameters (bounds, update deltas, thresholds, uncertainty
    decay) are read from ``params.WELFARE_LAYER_PARAMS`` at construction
    time so that robustness sweeps can vary them globally. The two most
    commonly-swept knobs (``default_vot``, ``learning_rate``) may still be
    passed as constructor arguments for backward compatibility; when they
    are omitted (``None``), the corresponding entry of
    ``params.WELFARE_LAYER_PARAMS`` is used.
    """

    def __init__(
        self,
        default_vot: Optional[float] = None,
        learning_rate: Optional[float] = None,
        config: Optional[Dict[str, float]] = None,
    ):
        cfg = dict(params.WELFARE_LAYER_PARAMS)
        if config:
            cfg.update(config)

        self.default_vot = default_vot if default_vot is not None else cfg["default_vot"]
        self.learning_rate = learning_rate if learning_rate is not None else cfg["tce_learning_rate"]

        # Snapshot all remaining knobs so later edits to
        # WELFARE_LAYER_PARAMS do not accidentally mutate an
        # already-constructed TCE.
        self.vot_lower = float(cfg["vot_lower"])
        self.vot_upper = float(cfg["vot_upper"])
        self.vot_dissatisfied_threshold = float(cfg["vot_dissatisfied_threshold"])
        self.vot_satisfied_threshold = float(cfg["vot_satisfied_threshold"])
        self.vot_long_trip_min = float(cfg["vot_long_trip_min"])
        self.vot_increase_rate = float(cfg["vot_increase_rate"])
        self.vot_decrease_rate = float(cfg["vot_decrease_rate"])
        self.uncertainty_initial = float(cfg["uncertainty_initial"])
        self.uncertainty_floor = float(cfg["uncertainty_floor"])
        self.uncertainty_decay_numerator = float(cfg["uncertainty_decay_numerator"])
        self.fallback_speed_kmh = float(cfg["fallback_speed_kmh"])
        self.fallback_cost_per_km = float(cfg["fallback_cost_per_km"])

        # Per-user state: {user_id: {"vot": float, "avg_travel_cost": float,
        # "avg_travel_time_hr": float, "avg_monetary_cost": float, "n": int}}
        # avg_travel_time_hr and avg_monetary_cost are running EMAs (same α)
        # used by ``estimate_travel_cost`` to derive an empirical-Bayes
        # anchor from avg_travel_cost.
        self.user_estimates: Dict[int, Dict[str, float]] = {}

    def update(self, feedback: ContinuousFeedback, travel_time_min: float, monetary_cost: float) -> None:
        """Update user travel cost estimates from a feedback observation.

        Implements app:tce of the manuscript: EMA-smooth the realised
        generalised cost reported by the user, then apply an asymmetric
        multiplicative update to v̂_i (5 % up, 3 % down) driven by the
        sign of the residual between observed and predicted travel
        cost under Eq. 7.

        Three points worth flagging (they differ from an earlier
        implementation that kept VOT learning essentially dormant at
        this horizon):

        (a) The EMA on ``avg_travel_cost`` is driven by
            ``feedback.realized_travel_cost`` — the *actual* cost the
            agent paid, using its true v_i, β^time, β^cost per Eq. 7 —
            rather than a cost re-synthesised with the TCE's own prior.

        (b) The asymmetric multiplicative VOT update is triggered by
            the **residual** between observed and predicted travel
            cost (Eq. 7 using the current v̂_i, with β = 1 since only
            v is learned — consistent with app:tce).  This is the
            paper's "dissatisfaction with a distant trip → v too low"
            heuristic, expressed in the variable the RS can actually
            measure.  In this simulation the satisfaction signal alone
            is a poor trigger because only *accepted* recommendations
            produce feedback, which biases satisfaction upward; the
            residual does not suffer from this truncation because it
            compares apples-to-apples (observed vs predicted cost on
            the same accepted trip) and captures the systematic bias
            between the TCE's prior and the user's true valuation.
            The 5 % / 3 % asymmetry (loss-aversion) is preserved.

        (c) The step size is scaled by
            ``min(1, |residual| / predicted_cost)`` so that a single
            noisy observation can move v̂ by at most the full 5 %
            (or 3 %) step, and tiny residuals produce tiny moves.

        Notes
        -----
        ``vot_dissatisfied_threshold``, ``vot_satisfied_threshold`` and
        ``vot_long_trip_min`` are retained as configuration fields for
        backward compatibility but are no longer used by this update
        rule.  The paper-relevant knobs are ``vot_increase_rate``,
        ``vot_decrease_rate``, ``vot_lower``, ``vot_upper``, and the
        EMA ``learning_rate``.
        """
        state = self.user_estimates.setdefault(
            feedback.user_id,
            {
                "vot": self.default_vot,
                "avg_travel_cost": 0.0,
                "avg_travel_time_hr": 0.0,
                "avg_monetary_cost": 0.0,
                "n": 0,
            },
        )
        state["n"] += 1
        alpha = self.learning_rate
        travel_time_hr = travel_time_min / 60.0

        # EMA of the realised generalised cost reported by the user plus
        # matching EMAs of the trip's τ and κ.  Keeping all three with a
        # common α makes the implied empirical VOT
        # (avg_gc − avg_κ) / avg_τ well-defined, which ``estimate_travel_cost``
        # then uses as a shrinkage anchor.  Initialise on first obs.
        realized_gc = float(feedback.realized_travel_cost)
        if state["n"] == 1:
            state["avg_travel_cost"] = realized_gc
            state["avg_travel_time_hr"] = travel_time_hr
            state["avg_monetary_cost"] = float(monetary_cost)
        else:
            state["avg_travel_cost"] = (
                (1.0 - alpha) * state["avg_travel_cost"] + alpha * realized_gc
            )
            state["avg_travel_time_hr"] = (
                (1.0 - alpha) * state["avg_travel_time_hr"] + alpha * travel_time_hr
            )
            state["avg_monetary_cost"] = (
                (1.0 - alpha) * state["avg_monetary_cost"] + alpha * float(monetary_cost)
            )

        # Residual-based asymmetric VOT update.
        if travel_time_hr <= 0:
            return
        predicted_cost = state["vot"] * travel_time_hr + monetary_cost
        residual = realized_gc - predicted_cost
        if residual == 0.0:
            return
        scale = min(1.0, abs(residual) / max(1e-3, predicted_cost))

        if residual > 0:
            # TCE under-predicted cost → true v > v̂ → bump up (loss-averse).
            state["vot"] = min(
                self.vot_upper,
                state["vot"] * (1 + self.vot_increase_rate * scale),
            )
        else:
            # TCE over-predicted cost → true v < v̂ → bump down (smaller step).
            state["vot"] = max(
                self.vot_lower,
                state["vot"] * (1 - self.vot_decrease_rate * scale),
            )

    def estimate_travel_cost(
        self,
        user_id: int,
        distance_km: float,
        avg_speed_kmh: Optional[float] = None,
        cost_per_km: Optional[float] = None,
    ) -> float:
        """Estimate generalised travel cost for a user-place pair.

        Returns Ĉ^travel_{ik} ≈ v̂_i · τ_{ik} + κ_{ik}, consistent with
        Eq. 7 of the manuscript (β̂ terms absorbed into v̂ because only
        v_i is updated online; see ``update``).

        When the user has accumulated enough feedback, we shrink the
        analytical estimate toward the empirical anchor implied by the
        EMAs of realised cost / travel time / monetary cost tracked in
        ``update``.  That uses ``avg_travel_cost`` (hence ``learning_rate``)
        to correct for any residual bias in v̂.  Shrinkage weight
        ``w(n) = min(0.5, n / 20)`` caps the empirical influence at 50 %
        and keeps the prior (v̂) driving when feedback is sparse.
        """
        state = self.user_estimates.get(user_id)
        speed = avg_speed_kmh if avg_speed_kmh is not None else self.fallback_speed_kmh
        per_km = cost_per_km if cost_per_km is not None else self.fallback_cost_per_km
        travel_time_hr = distance_km / max(1.0, speed)
        monetary_cost = distance_km * per_km

        if state is None or state["n"] == 0:
            return self.default_vot * travel_time_hr + monetary_cost

        vot = state["vot"]
        analytical = vot * travel_time_hr + monetary_cost

        # Empirical-Bayes anchor from the EMAs.  Skip when we haven't
        # built up enough history or when avg_travel_time_hr is too
        # small to invert safely.
        n = int(state["n"])
        if n < 3 or state["avg_travel_time_hr"] <= 1e-4:
            return analytical

        implied_vot = (
            state["avg_travel_cost"] - state["avg_monetary_cost"]
        ) / state["avg_travel_time_hr"]
        # Clamp the implied VOT to the same bounds as v̂ to guard against
        # noise blowing up the estimate.
        implied_vot = max(self.vot_lower, min(self.vot_upper, implied_vot))
        empirical = implied_vot * travel_time_hr + monetary_cost
        w = min(0.5, n / 20.0)
        return (1.0 - w) * analytical + w * empirical

    def get_user_vot(self, user_id: int) -> float:
        """Return estimated VOT for a user."""
        state = self.user_estimates.get(user_id)
        return state["vot"] if state else self.default_vot

    def get_user_uncertainty(self, user_id: int) -> float:
        """Return uncertainty (σ) for user's utility estimate.

        Decreases with more observations. Used in PUP computation.

        sigma(n=0) = uncertainty_initial
        sigma(n>0) = max(uncertainty_floor,
                         uncertainty_decay_numerator / sqrt(n))

        Baseline: uncertainty_initial = 1.0,
                  uncertainty_floor = 0.15,
                  uncertainty_decay_numerator = 1.0.
        """
        state = self.user_estimates.get(user_id)
        if state is None or state["n"] == 0:
            return self.uncertainty_initial
        return max(
            self.uncertainty_floor,
            self.uncertainty_decay_numerator / math.sqrt(state["n"]),
        )


# ── PUP and RM computation ───────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_pup(
    v_hat_act: float,
    c_hat_travel: float,
    sigma_i: float,
) -> float:
    """Compute Positive Utility Probability.

    PUP = Pr[U_{ik} ≥ 0] ≈ Φ((V̂^act - Ĉ^travel) / σ_i)

    Parameters
    ----------
    v_hat_act : float
        Estimated activity benefit.
    c_hat_travel : float
        Estimated travel cost.
    sigma_i : float
        User-specific uncertainty parameter.

    Returns
    -------
    float
        Probability in [0, 1] that net utility is non-negative.
    """
    if sigma_i <= 0:
        return 1.0 if v_hat_act >= c_hat_travel else 0.0
    z = (v_hat_act - c_hat_travel) / sigma_i
    return _normal_cdf(z)


def compute_expected_regret(
    v_hat_act: float,
    c_hat_travel: float,
    sigma_i: float,
    best_known_utility: float = 0.0,
) -> float:
    """Compute expected regret for a recommendation.

    E[R_{ik}] = E[max(0, U* - U_{ik})]

    For a normal approximation:
    E[R] = σ · φ(z) + (best - μ) · Φ(-z)
    where μ = V̂^act - Ĉ^travel, z = (μ - best) / σ

    Parameters
    ----------
    v_hat_act : float
        Estimated activity benefit.
    c_hat_travel : float
        Estimated travel cost.
    sigma_i : float
        User-specific uncertainty.
    best_known_utility : float
        Best alternative utility (0 = doing nothing).

    Returns
    -------
    float
        Expected regret, non-negative.
    """
    mu = v_hat_act - c_hat_travel
    if sigma_i <= 1e-8:
        return max(0.0, best_known_utility - mu)

    z = (mu - best_known_utility) / sigma_i
    # φ(z) = standard normal pdf
    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    # Φ(-z)
    cdf_neg_z = _normal_cdf(-z)
    return sigma_i * phi_z + (best_known_utility - mu) * cdf_neg_z


def estimate_activity_utility_from_score(
    score: float,
    leisure_subtype: str,
    mode: str = DEFAULT_SCORE_TO_UTILITY_MODE,
    score_scale: float = DEFAULT_SCORE_UTILITY_SCALE,
) -> float:
    """Map an RS score into the welfare layer's activity-utility scale.

    Modes
    -----
    - ``identity``: legacy behavior, v_hat_act = score
    - ``subtype_base_plus_score``: add the subtype's assumed base activity
      utility to the RS score so the welfare layer operates on the same
      utility scale used elsewhere in the ABM.

    Set ``mode='identity'`` to revert to the previous behavior.
    """
    bounded_score = max(0.0, float(score))
    if mode == LEGACY_SCORE_TO_UTILITY_MODE:
        return bounded_score
    scaled_score = score_scale * bounded_score
    if mode == DEFAULT_SCORE_TO_UTILITY_MODE:
        base_utility = params.LEISURE_SEGMENTS.get(leisure_subtype, {}).get("activity_utility", 0.0)
        return base_utility + scaled_score
    raise ValueError(f"Unknown score-to-utility mode: {mode}")


# ── Welfare-aware orchestrator ───────────────────────────────────────────────

@dataclass
class WelfareAnnotatedRecommendation:
    """A recommendation annotated with welfare metrics."""

    recommendation: Recommendation
    v_hat_act: float
    c_hat_travel: float
    pup: float
    expected_regret: float
    passed_pup: bool
    passed_rm: bool


class WelfareAwareOrchestrator:
    """Wraps LeisureRSOrchestrator with PUP/RM gating.

    Treatment modes:
    - "standard": pass-through (no welfare filtering)
    - "pup": filter recommendations where PUP < α
    - "rm": filter recommendations where E[R] > ε
    - "pup_rm": apply both constraints
    """

    def __init__(
        self,
        base_orchestrator: LeisureRSOrchestrator,
        travel_cost_estimator: TravelCostEstimator,
        mode: str = "standard",
        pup_alpha: float = 0.6,
        rm_epsilon: float = 0.3,
        coord_scale_km: float = 1.0,
        score_to_utility_mode: str = DEFAULT_SCORE_TO_UTILITY_MODE,
        score_utility_scale: float = DEFAULT_SCORE_UTILITY_SCALE,
    ):
        self.base = base_orchestrator
        self.tce = travel_cost_estimator
        self.mode = mode
        self.pup_alpha = pup_alpha
        self.rm_epsilon = rm_epsilon
        self.coord_scale_km = coord_scale_km
        self.score_to_utility_mode = score_to_utility_mode
        self.score_utility_scale = score_utility_scale
        # Feedback history for continuous learning
        self.feedback_history: List[ContinuousFeedback] = []

    def recommend(
        self,
        user: UserContext,
        leisure_subtype: str,
        top_k_per_system: int = 5,
    ) -> Dict[str, List[WelfareAnnotatedRecommendation]]:
        """Get welfare-annotated recommendations.

        Returns recommendations from the base orchestrator, each annotated
        with PUP and RM scores, and filtered according to the active mode.
        """
        base_recs = self.base.recommend(user, leisure_subtype, top_k_per_system)
        sigma_i = self.tce.get_user_uncertainty(user.user_id)

        annotated: Dict[str, List[WelfareAnnotatedRecommendation]] = {}
        for source, recs in base_recs.items():
            annotated_list = []
            for rec in recs:
                # Estimate V̂^act on the same scale as ABM activity utility.
                v_hat_act = estimate_activity_utility_from_score(
                    rec.score,
                    leisure_subtype,
                    mode=self.score_to_utility_mode,
                    score_scale=self.score_utility_scale,
                )

                # Estimate Ĉ^travel
                dist_km = euclidean_distance_km(user.location, rec.place.location) * self.coord_scale_km
                c_hat_travel = self.tce.estimate_travel_cost(user.user_id, dist_km)

                pup = compute_pup(v_hat_act, c_hat_travel, sigma_i)
                e_regret = compute_expected_regret(v_hat_act, c_hat_travel, sigma_i)

                passed_pup = pup >= self.pup_alpha
                passed_rm = e_regret <= self.rm_epsilon

                annotated_list.append(
                    WelfareAnnotatedRecommendation(
                        recommendation=rec,
                        v_hat_act=v_hat_act,
                        c_hat_travel=c_hat_travel,
                        pup=pup,
                        expected_regret=e_regret,
                        passed_pup=passed_pup,
                        passed_rm=passed_rm,
                    )
                )

            # Apply welfare filter
            if self.mode == "pup":
                annotated_list = [a for a in annotated_list if a.passed_pup]
            elif self.mode == "rm":
                annotated_list = [a for a in annotated_list if a.passed_rm]
            elif self.mode == "pup_rm":
                annotated_list = [a for a in annotated_list if a.passed_pup and a.passed_rm]
            # "standard" → no filtering

            annotated[source] = annotated_list

        return annotated

    def record_continuous_feedback(self, feedback: ContinuousFeedback, travel_time_min: float, monetary_cost: float) -> None:
        """Record continuous feedback and update all learning components."""
        self.feedback_history.append(feedback)

        # Update travel cost estimator
        self.tce.update(feedback, travel_time_min, monetary_cost)

    def record_feedback(self, user_id: int, source: str, place_id: str, liked: bool, feedback_strength: float = 1.0) -> None:
        """Backward-compatible binary feedback passthrough."""
        self.base.record_feedback(
            user_id=user_id,
            source=source,
            place_id=place_id,
            liked=liked,
            feedback_strength=feedback_strength,
        )
