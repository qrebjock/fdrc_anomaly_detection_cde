from fdrc.filters import (
    FixedThresholdState,
    FixedThresholdFilter,
    LORDState,
    LORDFilter,
    SAFFRONState,
    SAFFRONFilter,
    DecaySAFFRONFilter,
    DecayLORDFilter,
    ADDISFilter,
    ADDISState,
)

from fdrc import make_gaussian_hypotheses


def test_filter(filter_type, state_type, filter_args):
    dataset = make_gaussian_hypotheses(
        size=1000,
        anomaly_ratio=0.01,
        signal_strength=3
    )
    filter_instance = filter_type(**filter_args)
    labels, state = filter_instance.step(dataset["p_values"])
    assert len(labels) == 1000
    assert type(state) == state_type


def run_all_unit_tests():

    print("Testing Fixed Threshold Filter...")
    test_filter(
        FixedThresholdFilter,
        FixedThresholdState,
        filter_args={"threshold": 0.1},
    )
    print("Done.")

    print("Testing Lord Filter...")
    test_filter(
        LORDFilter,
        LORDState,
        filter_args={"fdr_target": 0.05},
    )
    print("Done.")

    print("Testing Decay Lord Filter...")
    test_filter(
        DecayLORDFilter,
        LORDState,
        filter_args={"fdr_target": 0.05},
    )
    print("Done.")

    print("Testing Saffron Filter...")
    test_filter(
        SAFFRONFilter,
        SAFFRONState,
        filter_args={"fdr_target": 0.05},
    )
    print("Done.")

    print("Testing Decay Saffron Filter...")
    test_filter(
        DecaySAFFRONFilter,
        SAFFRONState,
        filter_args={"fdr_target": 0.05},
    )
    print("Done.")

    print("Testing ADDIS Filter...")
    test_filter(
        ADDISFilter,
        ADDISState,
        filter_args={"fdr_target": 0.05},
    )
    print("Done.")


if __name__ == "__main__":
    run_all_unit_tests()
