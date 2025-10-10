import torch
import torch.nn as nn

from src.model.DomainMistralModel import DomainMistralForCausalLM


class TokenTrace:
    def __init__(self, input_tokens=None, masks_all_layers=None, tokens_all_layers=None,
                 output_tokens=None):
        """
        Token trace information.
        Args:
            input_tokens: List of tokens in the input text
            masks_all_layers: List of masks for all layers
            tokens_all_layers: List of tokens for all layers
            output_tokens: List of tokens in the output text
        """
        self.input_tokens = input_tokens
        self.masks_all_layers = masks_all_layers
        self.tokens_all_layers = tokens_all_layers
        self.output_tokens = output_tokens


class DomainMistralTokenMonitor:
    def __init__(self, model: DomainMistralForCausalLM, tokenizer):
        """
        Register a hook on model to monitor the trace of token embeddings.
        Args:
            model: DomainMistralForCausalLM
            tokenizer: tokenizer used to decode the tokens
        """
        self.model = model
        self.tokenizer = tokenizer

        # trace information
        # List of tokens in the input text
        self.token_traces = []

        self.register_hooks()

    """
    Hooks for the model to monitor the trace of token embeddings.
    """
    def input_text_hook(self, model: DomainMistralForCausalLM, input, output):
        input_text = [self.tokenizer.convert_ids_to_tokens(sentence) for sentence in input[0]]
        for sentence in input_text:
            self.token_traces.append(TokenTrace(input_tokens=sentence))

    def output_text_hook(self, model: DomainMistralForCausalLM, input, output):
        output_text = [self.tokenizer.convert_ids_to_tokens(sentence.argmax(dim=-1)) for sentence in output.logits]
        for i, sentence in enumerate(output_text):
            self.token_traces[i].output_tokens = sentence[-1]



    def register_hooks(self):
        self.model.register_forward_hook(self.input_text_hook)
        self.model.register_forward_hook(self.output_text_hook)

    def unregister_hooks(self):
        self.model.unregister_forward_hook(self.input_text_hook)
        self.model.unregister_forward_hook(self.output_text_hook)


