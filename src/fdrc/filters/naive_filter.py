from dataclasses import dataclass, field
from typing import List
from .fdrc_filter import FDRState, FDRFilter


@dataclass
class FixedThresholdState(FDRState):
    threshold: float
    num_tests: int = 0
    num_discoveries: int = 0
    threshold_history: List[float] = field(default_factory=list)


class FixedThresholdFilter(FDRFilter):
    """
    Naive Online FDR control filter class.

    Effectively behaves as a fixed threshold filter and does NOT control FDR.
    """
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def initial_state(self) -> FixedThresholdState:
        return FixedThresholdState(threshold=self.threshold)

    def single_step_update(self, score: float, state: FixedThresholdState) -> float:
        """
        Calculate current anomaly label and update the state.
        """
        label = float(score < state.threshold)
        state.num_tests += 1
        state.num_discoveries += int(label)
        state.threshold_history.append(state.threshold)
        return label
