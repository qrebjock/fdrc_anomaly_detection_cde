import numpy as np

from typing import List
from bisect import bisect_right
from dataclasses import dataclass, field
from .fdrc_filter import FDRState, FDRFilter


@dataclass
class ADDISState(FDRState):
    gamma_vec: List[float]
    num_tests: int = 0
    candidate_event_time: List[int] = field(default_factory=list)
    times_of_rejection: List[int] = field(default_factory=list)
    threshold_history: List[float] = field(default_factory=list)

    def gamma(self, index: int) -> float:
        if index > len(self.gamma_vec):
            return 0
        return self.gamma_vec[index - 1]


class ADDISFilter(FDRFilter):
    """
    ADDIS online FDR Control method with fixed lambda (candidacy threshold)
    and tau (discarding threshold).
    Implements a slight modification of the algorithm presented in the paper
    https://arxiv.org/abs/1802.09098, controlling the smoothed FDR
    E[V / (R + eta)] for some eta > 0, where V and R are decayed over time by
    a factor delta.
    This is a strict generalization of the DecaySAFFRONFilter controller.
    """

    def __init__(
        self,
        fdr_target: float,
        candidacy_threshold: float = 0.25,
        discarding_threshold: float = 0.5,
        gamma_exponent: float = 1.6,
        gamma_size: int = 100,
        delta: float = 1,
        eta: float = 1
    ) -> None:
        assert (
            0 <= fdr_target <= 1
        ), f"The FDR target must be between 0 and 1, found {fdr_target}."
        self.fdr_target = fdr_target

        assert (
            0 < candidacy_threshold < 1
        ), f"The candidacy_threshold must be between 0 and 1, found {candidacy_threshold}."
        self.candidacy_threshold = candidacy_threshold

        assert (
            candidacy_threshold < discarding_threshold <= 1
        ), f"The candidacy_threshold must be less than discarding_threshold"
        self.discarding_threshold = discarding_threshold

        assert gamma_exponent > 1
        self.gamma_exponent = gamma_exponent

        assert gamma_size > 1
        self.gamma_size = gamma_size

        gamma_vec = np.power(
            np.arange(start=1, stop=gamma_size + 1), -np.float(gamma_exponent)
        )
        gamma_vec = gamma_vec / np.float(np.sum(gamma_vec))
        self.gamma_vec = gamma_vec

        assert (
            0 <= delta <= 1
        ), f"The discount factor delta must be between 0 and 1, found {delta}."
        self.delta = delta

        assert eta > 0, f"The FDR offset eta must be positive, found {eta}"
        self.eta = eta

    def initial_state(self) -> ADDISState:
        state = ADDISState(num_tests=0, gamma_vec=list(self.gamma_vec))
        return state

    def single_step_update(self, score: float, state: ADDISState) -> float:
        """
        Calculate current anomaly label and update the state.
        """
        state.num_tests += 1

        # Threshold and label
        threshold = self._get_new_threshold(state=state)
        state.threshold_history.append(threshold)
        label = float(score < threshold)

        # Check if score is candidate
        if self.candidacy_threshold < score <= self.discarding_threshold:
            state.candidate_event_time.append(state.num_tests)

        # Update candidates count and time of rejection
        if label:
            state.times_of_rejection.append(state.num_tests)

        return label

    def _get_new_threshold_old(self, state: ADDISState) -> float:
        """
        This is the original _get_new_threshold method. It runs very slow due to running
        "get_candidates_since_rejection" subroutine for all previous candidates for all
        rejections. This makes O(MK) complexity for a single point. For a time series of
        N points, M-->N and the total complexity becomes O(N^2K).

        I replaced this function with a new one, that uses bisection method. It is
        significantly faster and allows us testing with more and longer time series.
        I want to keep this original version for reference. (Baris)

        PS: we can make it even faster with storing additional table in the state.
        """
        def _get_candidates_since_rejection(tau: int) -> int:
            candidates_mask = np.greater(state.candidate_event_time, tau)
            return np.count_nonzero(candidates_mask)

        def _get_gamma_vec_val(index: int) -> float:
            if index > len(state.gamma_vec):
                return 0
            else:
                return state.gamma_vec[index - 1]

        num_tests = state.num_tests

        base_term = self.eta * max(
            _get_gamma_vec_val(1 + len(state.candidate_event_time)),
            1 - self.delta,
        )

        rejections_term = 0
        for tau in state.times_of_rejection:
            rejections_term += self.delta ** (
                num_tests - tau
            ) * _get_gamma_vec_val(1 + _get_candidates_since_rejection(tau))

        threshold = (
            self.fdr_target
            # * (self.discarding_threshold - self.candidacy_threshold)
            * (self.candidacy_threshold)
            * (base_term + rejections_term)
        )

        return min(self.candidacy_threshold, threshold)

    def _get_new_threshold(self, state: ADDISState) -> float:
        num_tests = state.num_tests

        base_term = self.eta * max(
            state.gamma(1+len(state.candidate_event_time)),
            1 - self.delta
        )

        rejections_term, low, high = 0, 0, len(state.candidate_event_time)
        for tau in state.times_of_rejection:
            # Get number of candidates since rejection time tau
            low = bisect_right(state.candidate_event_time, tau, lo=low)
            candidates_since_tau = high - low

            gamma = state.gamma(candidates_since_tau+1)
            rejections_term += self.delta ** (num_tests - tau) * gamma

        threshold = (
            self.fdr_target * self.candidacy_threshold * (base_term + rejections_term)
            # * (self.discarding_threshold - self.candidacy_threshold)
        )

        return min(self.candidacy_threshold, threshold)