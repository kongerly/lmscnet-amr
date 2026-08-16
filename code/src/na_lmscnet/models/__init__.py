"""Model implementations for reproducible AMR experiments."""

from torch import nn

from na_lmscnet.models.cldnn import CLDNN
from na_lmscnet.models.cnn2 import CNN2
from na_lmscnet.models.extended_baselines import (
    MCLDNN,
    SEMSFN1D,
    MobileNetV2_1D,
    ResNet1DMACMatched,
)
from na_lmscnet.models.final_lmscnet import (
    AFNetAdaptation,
    LMSCNetS0,
    LMSCNetS1,
    LMSCNetS1Static,
    LMSCNetS1WideStatic,
    LMSCNetS2,
    LMSCNetS2Mean,
    LMSCNetS2Shuffled,
    SKNet1DAdaptation,
    shuffled_gate_weights,
)
from na_lmscnet.models.na_lmscnet import (
    NA_LMSCNet,
    NALMSCNet,
    NALMSCNetFixedAverage,
    NALMSCNetWithoutMultiScale,
    NALMSCNetWithoutSNRAuxiliary,
)
from na_lmscnet.models.resnet1d import ResNet1D
from na_lmscnet.models.source_vtcnn2 import SourceVTCNN2


def build_model(
    name: str,
    *,
    num_classes: int,
    dropout: float,
    expansion: float = 1.25,
    kernel: int | None = None,
    permutation_seed: int = 13,
) -> nn.Module:
    """Build a baseline by its configuration name."""

    constructors = {
        "cnn2": CNN2,
        "cldnn": CLDNN,
        "mcldnn": MCLDNN,
        "mobilenetv2_1d": MobileNetV2_1D,
        "na_lmscnet": NALMSCNet,
        "na_lmscnet_fixed_average": NALMSCNetFixedAverage,
        "na_lmscnet_wo_multi_scale": NALMSCNetWithoutMultiScale,
        "na_lmscnet_wo_snr_auxiliary": NALMSCNetWithoutSNRAuxiliary,
        "resnet1d": ResNet1D,
        "resnet1d_macs": ResNet1DMACMatched,
        "se_msfn_1d": SEMSFN1D,
        "lmscnet_s1_static": LMSCNetS1Static,
        "lmscnet_s1_wide_static": LMSCNetS1WideStatic,
        "lmscnet_s2_mean": LMSCNetS2Mean,
        "lmscnet_s2_shuffled": LMSCNetS2Shuffled,
        "sknet_1d_adaptation": SKNet1DAdaptation,
        "afnet_adaptation": AFNetAdaptation,
    }
    if name in {"lmscnet_s0_k3", "lmscnet_s0_k7", "lmscnet_s0_k15", "lmscnet_s0_wide"}:
        if kernel is None:
            raise ValueError(f"{name} requires an explicit kernel")
        expected_kernel = {
            "lmscnet_s0_k3": 3,
            "lmscnet_s0_k7": 7,
            "lmscnet_s0_k15": 15,
            "lmscnet_s0_wide": 7,
        }[name]
        if kernel != expected_kernel:
            raise ValueError(f"{name} is frozen to kernel {expected_kernel}")
        return LMSCNetS0(
            kernel=kernel,
            num_classes=num_classes,
            dropout=dropout,
            expansion=expansion,
        )
    if name == "lmscnet_s1":
        return LMSCNetS1(num_classes=num_classes, dropout=dropout, expansion=expansion)
    if name == "lmscnet_s2":
        return LMSCNetS2(num_classes=num_classes, dropout=dropout, expansion=expansion)
    if name == "lmscnet_s1_static":
        return LMSCNetS1Static(num_classes=num_classes, dropout=dropout, expansion=expansion)
    if name == "lmscnet_s1_wide_static":
        return LMSCNetS1WideStatic(num_classes=num_classes, dropout=dropout, expansion=expansion)
    if name == "lmscnet_s2_mean":
        return LMSCNetS2Mean(num_classes=num_classes, dropout=dropout, expansion=expansion)
    if name == "lmscnet_s2_shuffled":
        return LMSCNetS2Shuffled(
            num_classes=num_classes,
            dropout=dropout,
            expansion=expansion,
            permutation_seed=permutation_seed,
        )
    if name == "sknet_1d_adaptation":
        return SKNet1DAdaptation(num_classes=num_classes, dropout=dropout, expansion=expansion)
    if name == "afnet_adaptation":
        return AFNetAdaptation(num_classes=num_classes, dropout=dropout, expansion=expansion)
    try:
        constructor = constructors[name]
    except KeyError as error:
        raise ValueError(f"Unsupported model name: {name}") from error
    return constructor(num_classes=num_classes, dropout=dropout)


__all__ = [
    "CLDNN",
    "CNN2",
    "LMSCNetS0",
    "LMSCNetS1",
    "LMSCNetS1Static",
    "LMSCNetS1WideStatic",
    "LMSCNetS2",
    "LMSCNetS2Mean",
    "LMSCNetS2Shuffled",
    "SKNet1DAdaptation",
    "AFNetAdaptation",
    "shuffled_gate_weights",
    "MCLDNN",
    "MobileNetV2_1D",
    "NA_LMSCNet",
    "NALMSCNet",
    "NALMSCNetFixedAverage",
    "NALMSCNetWithoutMultiScale",
    "NALMSCNetWithoutSNRAuxiliary",
    "ResNet1D",
    "ResNet1DMACMatched",
    "SEMSFN1D",
    "SourceVTCNN2",
    "build_model",
]
