from far2.config import ModelConfig
from far2.models.base import ModelOutput, ReasoningModel
from far2.models.far2 import FAR2Model
from far2.models.recurrent import RecurrentModel
from far2.models.transformer import TransformerModel


def build_model(cfg: ModelConfig) -> ReasoningModel:
    return {"transformer": TransformerModel,
            "recurrent": RecurrentModel,
            "far2": FAR2Model}[cfg.kind](cfg)


__all__ = ["ModelConfig", "ModelOutput", "ReasoningModel", "build_model",
           "TransformerModel", "RecurrentModel", "FAR2Model"]
