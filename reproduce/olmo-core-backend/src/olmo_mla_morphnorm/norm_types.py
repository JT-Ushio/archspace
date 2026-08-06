from olmo_core.config import StrEnum


class MLANormType(StrEnum):
    """QK normalization variants supported by MLA."""

    baseline = "baseline"
    materialize_norm = "materialize_norm"
    standard_qk_norm = "standard_qk_norm"
    morphnorm = "morphnorm"
