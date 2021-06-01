from .fdrc_filter import FDRFilter, FDRState
from .naive_filter import FixedThresholdState, FixedThresholdFilter
from .lord_filter import LORDState, LORDFilter, DecayLORDFilter
from .saffron_filter import SAFFRONState, SAFFRONFilter, DecaySAFFRONFilter
from .addis_filter import ADDISFilter, ADDISState


FILTERS = {
    "FixedThresholdFilter": FixedThresholdFilter,
    "LORDFilter": LORDFilter,
    "DecayLORDFilter": DecayLORDFilter,
    "SAFFRONFilter": SAFFRONFilter,
    "DecaySAFFRONFilter": DecaySAFFRONFilter,
    "ADDISFilter": ADDISFilter
}


def build_filter(name: str, params: dict):
    return FILTERS[name](**params)
