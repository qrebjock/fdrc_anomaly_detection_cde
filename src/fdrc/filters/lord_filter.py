import numpy as np

from dataclasses import dataclass, field
from typing import List
from .fdrc_filter import FDRState, FDRFilter


@dataclass
class LORDState(FDRState):

    wealth: float
    gamma_vec: List[float]
    num_tests: int = 0
    abstain: bool = False
    discounted_discoveries: float = 0.0
    first_rejection: int = np.inf
    times_of_rejection: List[int] = field(default_factory=list)
    wealth_history: List[float] = field(default_factory=list)
    threshold_history: List[float] = field(default_factory=list)

    def gamma(self, index: int) -> float:
        if index > len(self.gamma_vec):
            return 0
        return self.gamma_vec[index - 1]

    def reset(self, initial_wealth):
        self.wealth = initial_wealth
        self.num_tests = 0
        self.times_of_rejection = []
        self.first_rejection = np.inf
        self.abstain = False
        self.discounted_discoveries = 0.0


class LORDFilter(FDRFilter):
    """
    LORD online FDR controller.
    Implementation of the GAI++ rule first proposed here https://arxiv.org/abs/1502.06197
    and then extended notably in this paper https://arxiv.org/abs/1710.00499.

    Parameters
    ----------
    fdr_target : float
        The level at which the false discovery rate should be controlled.
    initial_wealth : float
        Initial wealth value. Should be less than fdr_target.
    gamma_size : int
        Size of the gamma buffer.
    delta : float
        Decay factor. Should be in [0, 1].
    epsilon_w : float
        Wealth threshold below which the algorithm starts abstaining.
    epsilon_r : float
        Threshold for the discounted discoveries R. During the abstention phase,
        if R goes above this threshold, the algorithm starts making decisions again.
    """

    def __init__(
        self,
        fdr_target: float,
        initial_wealth: float = None,
        gamma_size: int = 100,
        delta: float = 1.0,
        epsilon_w: float = 0.0,
        epsilon_r: float = 0.0,
    ):
        assert 0 <= fdr_target <= 1, \
            f"The FDR target must be between 0 and 1, found {fdr_target}."
        self.fdr_target = float(fdr_target)

        initial_wealth = initial_wealth if initial_wealth else fdr_target / 2
        assert 0 <= initial_wealth <= fdr_target, \
            f"The initial wealth must be non-negative and less than the FDR target."
        self.initial_wealth = float(initial_wealth)

        assert gamma_size >= 0, \
            f"Specify a non-negative value for the gamma buffer size"
        self.gamma_size = int(gamma_size)

        assert 0 <= delta <= 1, \
            f"The discount factor delta must be between 0 and 1, found {delta}."
        self.delta = float(delta)

        self.b0 = self.fdr_target - self.initial_wealth
        self.epsilon_r = float(epsilon_r)
        self.epsilon_w = float(epsilon_w)

    def _build_gamma_vector(self) -> List[float]:
        if self.gamma_size == 0:
            return []
        t = np.arange(1, self.gamma_size + 1)
        gamma_vec = np.log(np.maximum(2, t)) / (t * np.exp(np.sqrt(np.log(t))))
        gamma_vec /= gamma_vec.sum()
        return list(gamma_vec)

    def initial_state(self) -> LORDState:
        return LORDState(
            wealth=self.initial_wealth,
            gamma_vec=self._build_gamma_vector()
        )

    def single_step_update(self, score: float, state: LORDState) -> float:
        """
        Calculate current anomaly label and update the state.
        """
        if state.abstain and state.discounted_discoveries < self.epsilon_r:
            state.reset(self.initial_wealth)

        label = 0.0
        threshold = 0.0
        state.num_tests += 1

        if not state.abstain:
            threshold = self._compute_threshold(state)
            label = float(score < threshold)

        if label:
            state.times_of_rejection.append(state.num_tests)
            if np.isinf(state.first_rejection):
                state.first_rejection = state.num_tests

        state.wealth = self.delta * state.wealth + (
            (1 - self.delta)
            * self.initial_wealth
            * float(state.first_rejection >= state.num_tests)
        )

        if not state.abstain:
            if state.first_rejection == state.num_tests:
                psi = self.b0
            else:
                psi = self.fdr_target
            state.wealth += psi * label - threshold

        # label is False if abstained
        state.discounted_discoveries = (
            state.discounted_discoveries * self.delta + label
        )

        state.wealth_history.append(state.wealth)
        state.threshold_history.append(threshold)
        if state.wealth < self.epsilon_w:
            state.abstain = True

        return label

    def _compute_threshold(self, state: LORDState) -> float:
        t = state.num_tests
        tau = np.inf

        if len(state.times_of_rejection) > 0:
            tau = state.times_of_rejection[0]

        threshold = (
            state.gamma(t)
            * self.initial_wealth
            * self.delta ** (t - min(t, tau))
        )

        if not np.isinf(tau):
            threshold += (
                self.b0
                * state.gamma(t - tau)
                * self.delta ** (t - tau)
            )

        threshold += self.fdr_target * np.sum([
                state.gamma(t - t0) * self.delta ** (t - t0)
                for t0 in state.times_of_rejection[1:]
            ]
        )

        return threshold


class DecayLORDFilter(FDRFilter):
    """
    LORD online FDR controller adapted to avoid alpha-death using decaying
    memory. Controls E[V / (R + eta)] for some eta > 0, where V and R are
    decayed over time by a factor delta.

    Parameters
    ----------
    fdr_target : float
        The level at which the false discovery rate should be controlled.
    gamma_size : int
        Size of the gamma buffer.
    delta : float
        Decay factor. Should be in [0, 1].
    eta : float
        Offset term in the FDR expression
    """

    def __init__(
        self,
        fdr_target: float,
        gamma_size: int = 100,
        delta: float = 1,
        eta: float = 1,
    ):
        assert 0 <= fdr_target <= 1, \
            f"The FDR target must be between 0 and 1, found {fdr_target}."
        self.fdr_target = fdr_target

        assert gamma_size >= 0, \
            f"Specify a non-negative value for the gamma buffer size"
        self.gamma_size = gamma_size
        t = np.arange(1, self.gamma_size + 1)
        self.gamma_vec = np.log(np.maximum(2, t)) / (t * np.exp(np.sqrt(np.log(t))))
        self.gamma_vec /= self.gamma_vec.sum()

        assert 0 <= delta <= 1, \
            f"The discount factor delta must be between 0 and 1, found {delta}."
        self.delta = delta

        assert eta > 0, f"The FDR offset eta must be positive, found {eta}"
        self.eta = eta

    def initial_state(self) -> LORDState:
        return LORDState(
            wealth=None,
            gamma_vec=self.gamma_vec
        )

    def single_step_update(self, score: float, state: LORDState) -> float:
        """
        Calculate current anomaly label and update the state.
        """
        state.num_tests += 1
        threshold = self._compute_threshold(state)
        state.threshold_history.append(threshold)

        label = float(score < threshold)
        if label:
            state.times_of_rejection.append(state.num_tests)

        return label

    def _compute_threshold(self, state: LORDState) -> float:
        threshold = (
            self.eta
            * self.fdr_target
            * max(state.gamma(state.num_tests), 1 - self.delta)
        )

        threshold += self.fdr_target * np.sum([
                state.gamma(state.num_tests - t) * self.delta ** (state.num_tests - t)
                for t in state.times_of_rejection
            ]
        )

        return threshold
