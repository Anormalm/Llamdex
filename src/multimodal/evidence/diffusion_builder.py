from src.multimodal.evidence.base import EvidenceBuilder


class DiffusionFutureEvidenceBuilder(EvidenceBuilder):
    """
    Reserved interface for Plan 2.
    """

    def __init__(self, evidence_dim: int):
        super().__init__(evidence_dim=evidence_dim)

    def forward(self, inputs):
        raise NotImplementedError(
            "Diffusion-based evidence builder is reserved for Plan 2 and is intentionally not implemented in this run."
        )

