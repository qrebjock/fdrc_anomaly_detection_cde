import numpy as np

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, List
from .fdrc_filter import FDRState, FDRFilter


@dataclass
class SAFFRONState(FDRState):
    fdr_target: float
    candidacy_threshold: float
    gamma_vec: List[float]
    num_tests: int = 0
    candidate_event_time: List[float] = field(default_factory=list)
    times_of_rejection: List[int] = field(default_factory=list)
    threshold_history: List[float] = field(default_factory=list)

    def get_candidates_since_rejection(self, tau: int) -> int:
        candidates_mask = np.greater(self.candidate_event_time, tau)
        return np.count_nonzero(candidates_mask)

    def gamma(self, index: int) -> float:
        if index > len(self.gamma_vec):
            return 0
        return self.gamma_vec[index - 1]


class SAFFRONFilter(FDRFilter):
    """
    SAFFRON Online FDR Control method with fixed lambda
    Implements the algorithm in section 2.3 of https://arxiv.org/abs/1802.09098
    """

    def __init__(
        self,
        fdr_target: float,
        candidacy_threshold: float = 0.5,
        gamma_exponent: float = 1.6,
        gamma_size: int = 100
    ) -> None:
        """
        The SAFFRON online FDR control rule that generates dynamic thresholds and
        returns labels given scores.

        Parameters
        ----------
        fdr_target: float
            The target share of false discovery, in (0, 1)
        candidacy_threshold: float
            scores larger than the candidacy threshold are always treated as
            non-anomalous, in (0, 1). This is the lambda parameter in the paper.
        gamma_exponent: float
            Controls how aggressively we forget the history. Strictly greater than 1.
            This is the s parameter in the paper.
        gamma_size: int
            Maximum memory span, int greater than 1.
        """
        assert 1 > fdr_target > 0
        self.fdr_target = float(fdr_target)

        assert 0 < candidacy_threshold < 1
        self.candidacy_threshold = float(candidacy_threshold)

        assert gamma_exponent > 1
        self.gamma_exponent = float(gamma_exponent)

        assert gamma_size > 1
        self.gamma_size = int(gamma_size)

        self.initial_wealth = self.fdr_target / 2

    def _build_gamma_vec(self):
        gamma_vec = np.power(
            np.arange(start=1, stop=self.gamma_size + 1), - self.gamma_exponent
        )
        return list(gamma_vec / np.sum(gamma_vec))

    def initial_state(self) -> Dict:
        return SAFFRONState(
            fdr_target=self.fdr_target,
            candidacy_threshold=self.candidacy_threshold,
            gamma_vec=self._build_gamma_vec()
        )

    def single_step_update(self, score: float, state: SAFFRONState) -> float:
        """
        Calculate current anomaly label and update the state.
        """
        state.num_tests += 1

        # Check if score is candidate
        if score <= state.candidacy_threshold:
            state.candidate_event_time.append(state.num_tests)

        # Get threshold and update
        threshold = self._get_new_threshold(state)
        state.threshold_history.append(threshold)

        # Get a label
        label = float(score < threshold)

        # Update candidates count and time of rejection
        if label:
            state.times_of_rejection.append(state.num_tests)

        return label

    def _get_new_threshold_old(self, state: SAFFRONState) -> float:
        """
        This is the original _get_new_threshold method. It runs very slow due to running
        "get_candidates_since_rejection" subroutine for all previous candidates for all
        rejections. This makes O(MK) complexity for a single point. For a timeseries of
        N points, M-->N and the total complexity becomes O(N^2K).

        I replaced this function with a new one, that uses bisection method. It is
        significantly faster and allows us testing with more and longer timeseries.
        I want to keep this original version for reference. (Baris)

        PS: we can make it even faster with storing additional table in the state.
        """
        if state.num_tests == 1:
            return min(
                state.candidacy_threshold,
                (1 - state.candidacy_threshold)
                * state.gamma(index=1)
                * self.initial_wealth,
            )

        gamma_0 = state.gamma(state.num_tests - len(state.candidate_event_time))
        alpha_tilde = self.initial_wealth * gamma_0

        if len(state.times_of_rejection) > 0:
            tau_1 = state.times_of_rejection[0]
            candidates_since_tau = state.get_candidates_since_rejection(tau_1)
            gamma_1 = state.gamma(state.num_tests - tau_1 - candidates_since_tau)
            alpha_tilde += (state.fdr_target - self.initial_wealth) * gamma_1

        for j in np.arange(2, 1 + len(state.times_of_rejection)):
            tau_j = state.times_of_rejection[j - 1]
            candidates_since_tau = state.get_candidates_since_rejection(tau_j)
            gamma_j = state.gamma(state.num_tests - tau_j - candidates_since_tau)
            alpha_tilde += state.fdr_target * gamma_j

        return min(
            state.candidacy_threshold, (1 - state.candidacy_threshold) * alpha_tilde
        )

    def _get_new_threshold(self, state: SAFFRONState) -> float:
        if state.num_tests == 1:
            return min(
                state.candidacy_threshold,
                (1 - state.candidacy_threshold)
                * state.gamma(1)
                * self.initial_wealth,
            )

        gamma_0 = state.gamma(state.num_tests - len(state.candidate_event_time))
        alpha_tilde = self.initial_wealth * gamma_0

        low, high = 0, len(state.candidate_event_time)
        for i, tau in enumerate(state.times_of_rejection):
            # Get number of candidates since rejection time tau
            low = bisect_right(state.candidate_event_time, tau, lo=low)
            candidates_since_tau = high - low

            gamma_tau = state.gamma(state.num_tests - tau - candidates_since_tau)
            if i == 0:
                alpha_tilde += (state.fdr_target - self.initial_wealth) * gamma_tau
            else:
                alpha_tilde += state.fdr_target * gamma_tau

        return min(
            state.candidacy_threshold, (1 - state.candidacy_threshold) * alpha_tilde
        )


class DecaySAFFRONFilter(SAFFRONFilter):
    """
    SAFFRON Online FDR Control method with fixed lambda adapted to avoid
    alpha-death using decaying memory. Controls E[V / (R + eta)] for some
    eta > 0, where V and R are decayed over time by a factor delta.

    Parameters
    ----------
    fdr_target: float
        The target share of false discovery, in (0, 1)
    candidacy_threshold: float
        Scores larger than the candidacy threshold are always treated as
        non-anomalous, in (0, 1). This is the lambda parameter in the paper.
    gamma_vec_exponent: float
        Controls how aggressively we forget the history, float strictly greater
        than 1. This is the s parameter in the paper.
    gamma_vec_length: int
        Maximum memory span, int greater than 1.
    """
    __NAME__ = "Decay SAFFRON"

    def __init__(
        self,
        fdr_target: float,
        candidacy_threshold: float = 0.5,
        gamma_exponent: float = 1.6,
        gamma_size: int = 100,
        delta: float = 1.0,
        eta: float = 1.0,
    ):
        super().__init__(
            fdr_target=fdr_target,
            candidacy_threshold=candidacy_threshold,
            gamma_exponent=gamma_exponent,
            gamma_size=gamma_size,
        )
        assert (
            0 <= delta <= 1
        ), f"The discount factor delta must be between 0 and 1, found {delta}."
        self.delta = delta

        assert eta > 0, f"The FDR offset eta must be positive, found {eta}"
        self.eta = eta

    def single_step_update(self, score: float, state: SAFFRONState) -> float:
        """
        Calculate current anomaly label and update the state.
        """
        state.num_tests += 1

        # Threshold and label
        threshold = self._get_new_threshold(state)
        state.threshold_history.append(threshold)
        label = float(score < threshold)

        # Check if score is candidate
        if score <= self.candidacy_threshold:
            state.candidate_event_time.append(state.num_tests)

        # Update candidates count and time of rejection
        if label:
            state.times_of_rejection.append(state.num_tests)

        return label

    def _get_new_threshold_old(self, state: SAFFRONState) -> float:
        base_term = self.eta * max(
            state.gamma(state.num_tests - len(state.candidate_event_time)),
            1 - self.delta,
        )
        rejections_term = 0
        for tau in state.times_of_rejection:
            candidates_since_tau = state.get_candidates_since_rejection(tau)
            gamma_val = state.gamma(state.num_tests - tau - candidates_since_tau)
            rejections_term += self.delta ** (state.num_tests - tau) * gamma_val
        return (
            state.fdr_target
            * (1 - state.candidacy_threshold)
            * (base_term + rejections_term)
        )

    def _get_new_threshold(self, state: SAFFRONState) -> float:
        base_term = self.eta * max(
            state.gamma(state.num_tests - len(state.candidate_event_time)),
            1 - self.delta,
        )

        rejections_term, low, high = 0, 0, len(state.candidate_event_time)
        for tau in state.times_of_rejection:
            # Get number of candidates since rejection time tau
            low = bisect_right(state.candidate_event_time, tau, lo=low)
            candidates_since_tau = high - low

            gamma_val = state.gamma(state.num_tests - tau - candidates_since_tau)
            rejections_term += self.delta ** (state.num_tests - tau) * gamma_val

        return (
            state.fdr_target
            * (1 - state.candidacy_threshold)
            * (base_term + rejections_term)
        )