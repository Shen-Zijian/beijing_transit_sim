"""路径选择模型：确定性最短广义成本，或多项 Logit。

广义成本 (s 当量)::

    GC = beta_ivt * in_vehicle_time + beta_wait * expected_wait
       + beta_walk * walk_time + beta_transfer * num_transfers

Logit: P_i = exp(-theta * GC_i) / sum_j exp(-theta * GC_j)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .config import ChoiceConfig


class RouteChoice:
    name = "base"

    def __init__(self, cfg: ChoiceConfig):
        self.cfg = cfg

    def generalized_cost(self, route: dict) -> float:
        c = self.cfg
        return (c.beta_ivt * route.get("in_vehicle_time", 0.0)
                + c.beta_wait * route.get("expected_wait", 0.0)
                + c.beta_walk * route.get("walk_time", 0.0)
                + c.beta_transfer * route.get("num_transfers", 0))

    def costs(self, routes: Sequence[dict]) -> np.ndarray:
        return np.asarray([self.generalized_cost(r) for r in routes], dtype=float)

    def probabilities(self, routes: Sequence[dict]) -> np.ndarray:
        raise NotImplementedError

    def select(self, routes: Sequence[dict], rng: np.random.Generator, passenger=None, now: float = 0.0) -> Optional[dict]:
        if not routes:
            return None
        if len(routes) == 1:
            return routes[0]
        p = self.probabilities(routes)
        idx = int(rng.choice(len(routes), p=p))
        return routes[idx]


class ShortestChoice(RouteChoice):
    name = "shortest"

    def probabilities(self, routes: Sequence[dict]) -> np.ndarray:
        c = self.costs(routes)
        p = np.zeros(len(routes))
        p[int(np.argmin(c))] = 1.0
        return p

    def select(self, routes, rng, passenger=None, now=0.0):
        if not routes:
            return None
        c = self.costs(routes)
        return routes[int(np.argmin(c))]


class LogitChoice(RouteChoice):
    name = "logit"

    def probabilities(self, routes: Sequence[dict]) -> np.ndarray:
        c = self.costs(routes)
        u = -self.cfg.theta * (c - c.min())
        w = np.exp(u)
        return w / w.sum()


def build_choice_model(cfg: ChoiceConfig) -> RouteChoice:
    model = (cfg.model or "logit").lower()
    if model == "shortest":
        return ShortestChoice(cfg)
    if model == "logit":
        return LogitChoice(cfg)
    raise ValueError(f"未知的路径选择模型: {cfg.model}")
