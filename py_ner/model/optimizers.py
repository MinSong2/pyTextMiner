from transformers import PreTrainedModel
import torch
from typing import Union


def get_optimizer_with_weight_decay(model: PreTrainedModel,
                                    optimizer: torch.optim.Optimizer,
                                    learning_rate: Union[float, int],
                                    weight_decay: Union[float, int]) -> torch.optim.Optimizer:
    """
    Apply weight decay to all the network parameters but those called `bias` or  `LayerNorm.weight`.
    Args:
        model (`PreTrainedModel`): model to apply weight decay.
        optimizer (`torch.optim.Optimizer`): The optimizer to use during training.
        learning_rate (`float` or `int`): value of the learning rate to use during training.
        weight_decay (`float` or `int`): value of the weight decay to apply.

    Returns:
        optimizer (`torch.optim.Optimizer`): the optimizer instantiated with the selected
        learning rate and the parameters with and without weight decay.

    """
    no_decay = ["bias", "LayerNorm.weight"]
    params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nd = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [{"params": params, "weight_decay": weight_decay},
                                    {"params": params_nd, "weight_decay": 0.0}]

    return optimizer(optimizer_grouped_parameters, lr=learning_rate)