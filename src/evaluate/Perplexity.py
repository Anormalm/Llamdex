import torch
from typing import Callable
import inspect


class Perplexity:
    """
    Compute the perplexity given a callable model and input_ids (tokenized input).
    Modified from https://huggingface.co/docs/transformers/main/en/perplexity

    Args:
        max_length (int): The maximum length of the input.
        stride (int): The stride of the input.
        device (str): The device to use.
    """
    def __init__(self, max_length=512, stride=512, device="cuda"):
        self.max_length = max_length
        self.stride = stride
        self.device = device if torch.cuda.is_available() else "cpu"

    def check_model(self, model: Callable) -> None:
        """
        Check if the model is compatible with the Perplexity class.
        Args:
            model: A callable model.

        Raises:
            ValueError: If the model is not compatible.
        """
        if not isinstance(model, torch.nn.Module):
            raise ValueError("The model must be a torch.nn.Module.")
        if not hasattr(model, "forward"):
            raise ValueError("The model must have a forward function.")

        # check if model has a "labels" parameter in the forward function
        if "labels" not in inspect.signature(model.forward).parameters:
            print(inspect.signature(model.forward).parameters)
            raise ValueError("The model must have a 'labels' parameter in the forward function.")

    def compute(self, model: Callable, input_ids: torch.Tensor) -> float:
        """
        Compute the perplexity given a callable model and input_ids (tokenized input).
        Args:
            model: A callable model.
            input_ids: The tokenized input. The current implementation supports a single input.

        Returns: The perplexity of the input.
        """
        self.check_model(model)

        seq_len = input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
