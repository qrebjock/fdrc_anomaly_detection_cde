from typing import List, Optional, Tuple


class FDRState:
    pass


class FDRFilter:

    def initial_state(self) -> FDRState:
        raise NotImplementedError

    def step(
        self, prob_scores: List[float], state: Optional[FDRState] = None
    ) -> Tuple[List[float], FDRState]:
        """
        Produces a list of labels and an updated state given a starting state
        and a list of scores.

        Parameters
        ----------
        prob_scores: List[float]
            A list of scores assimilable to p-values in [0,1].
        state: FDRState
            The state

        Returns
        -------
        Tuple[List[float], FDRState]
        """
        state = state if state is not None else self.initial_state()
        labels = [self.single_step_update(score, state) for score in prob_scores]
        return labels, state

    def single_step_update(
        self, score: float, state: FDRState
    ) -> float:
        """
        Receives current score and state, returns the label and updates the state.

        Parameters
        ----------
        score: float
            A p-value in [0,1].
        state: FDRState
            A filter state

        Returns
        -------
        float:
            Anomaly label, 0.0 or 1.0
        """
        raise NotImplementedError
