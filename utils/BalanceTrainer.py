import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from peft import get_peft_model_state_dict
import torch
from torch import nn
from transformers import Trainer, PreTrainedModel
from transformers.modeling_utils import unwrap_model

WEIGHTS_NAME = "adapter_model.bin"


@dataclass
class LLMClassificationLabelSmoother:

    epsilon: float = 0.1  
    classification_alpha: float = 0.1  
    ignore_index: int = -100

    def __call__(self, model_output, labels, classification_index, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]

        if shift_labels:  
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        token_shape = labels.shape 

        smoothing_loss_func = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.epsilon) 
        normal_loss_func = nn.CrossEntropyLoss(reduction="mean")

        token_loss = (smoothing_loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
                      .view(token_shape)) 

        origin_loss = normal_loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1)) 

        classification_index = classification_index.view(
            (-1, 1)) 

        classification_loss = token_loss.gather(dim=1, index=classification_index).mean()  

        return self.classification_alpha * classification_loss + (1 - self.classification_alpha) * origin_loss


class BalanceTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.args.label_smoothing_factor is not None:
            self.label_smoother = LLMClassificationLabelSmoother(epsilon=self.args.label_smoothing_factor,
                                                                 classification_alpha=self.args.classification_alpha)
        else:
            self.label_smoother = None

    def compute_loss(self, model, inputs, return_outputs=False):

        if self.args.label_smoothing_factor is not None and "labels" in inputs:
            inputs_labels = inputs.pop("labels") 
            labels = inputs_labels[:, 1:] 
            label_smooth_index = inputs_labels[:, 0]
            inputs['input_ids'] = inputs['input_ids'][:, 1:] 

        else:
            labels = None

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None: 
            loss = self.label_smoother(outputs, labels, label_smooth_index, shift_labels=True) 
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):  
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not isinstance(self.model, PreTrainedModel):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(get_peft_model_state_dict(self.model, state_dict), os.path.join(output_dir, WEIGHTS_NAME))

            try:
                unwrap_model(self.model).peft_config.save_pretrained(output_dir)
            except AttributeError:
                unwrap_model(self.model).peft_config['default'].save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)