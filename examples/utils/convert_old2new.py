import json
import re

import torch
from safetensors.torch import load_file

from transformers import AutoTokenizer
from transformers.utils import logging
from src.small_doge.models.modeling_old_doge import DogeForCausalLM as OldDogeForCausalLM
from src.small_doge.models.modeling_doge import DogeForCausalLM
from src.small_doge.models.configuration_doge import DogeConfig


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

STATE_DICT_MAPPING = {
    # LM_head keys
    r"^lm_head.weight": r"lm_head.weight",

    # Model keys
    r"^model.final_layernorm.weight": r"model.final_layernorm.weight",
    r"^model.word_embed.weight": r"model.word_embed.weight",
    r"^model.rotary_emb.rotary_emb": r"model.rotary_emb.rotary_emb",
    r"^model.rotary_emb.original_inv_freq": r"model.rotary_emb.original_inv_freq",

    # Layers keys
    r"^model.layers.(\d+).pre_layernorm.weight": r"model.layers.\1.pre_layernorm.weight",
    r"^model.layers.(\d+).pre_residual.weight": r"model.layers.\1.pre_residual.weight",
    r"^model.layers.(\d+).post_layernorm.weight": r"model.layers.\1.post_layernorm.weight",
    r"^model.layers.(\d+).post_residual.weight": r"model.layers.\1.post_residual.weight",

    # Attention keys
    r"^model.layers.(\d+).self_attn.q_proj.weight": r"model.layers.\1.self_attn.q_proj.weight",
    r"^model.layers.(\d+).self_attn.k_proj.weight": r"model.layers.\1.self_attn.k_proj.weight",
    r"^model.layers.(\d+).self_attn.v_proj.weight": r"model.layers.\1.self_attn.v_proj.weight",
    r"^model.layers.(\d+).self_attn.A": r"model.layers.\1.self_attn.A",
    r"^model.layers.(\d+).self_attn.dt_proj.weight": r"model.layers.\1.self_attn.dt_proj.weight",
    r"^model.layers.(\d+).self_attn.o_proj.weight": r"model.layers.\1.self_attn.o_proj.weight",


    # MLP keys
    r"^model.layers.(\d+).feed_forward.gate_proj.weight": r"model.layers.\1.feed_forward.gate_proj.weight",
    r"^model.layers.(\d+).feed_forward.up_proj.weight": r"model.layers.\1.feed_forward.up_proj.weight",
    r"^model.layers.(\d+).feed_forward.down_proj.weight": r"model.layers.\1.feed_forward.down_proj.weight",
}


def map_old_key_to_new(old_key):
    """Map of a key of the original state dict to the equivalent key in HF format"""
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def convert_state_dict(original_state_dict: dict, config: DogeConfig):
    """Convert a state dict file, when a single `nn.Module` is never sharded in different files (usual case)."""
    new_dict = {}

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)

        new_dict[new_key] = tensor
    return new_dict


def validate_converted_model(
    old_model: OldDogeForCausalLM, new_model: DogeForCausalLM, tokenizer: AutoTokenizer
) -> None:
    """Validate the converted model returns the same output as the original model."""
    torch_device = "cpu"

    # Move models to device
    old_model = old_model.to(torch_device)
    new_model = new_model.to(torch_device)
    input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"].to(torch_device)

    # Assert model logits are close
    with torch.no_grad():
        old_model_logits = old_model(input_ids).logits
        new_model_logits = new_model(input_ids).logits
        print(f"old_model_logits: {old_model_logits}")
        print(f"new_model_logits: {new_model_logits}")
    if not torch.allclose(old_model_logits, new_model_logits, atol=1e-3):
        raise ValueError("The converted model did not return the same logits as the original model.")

    logger.info("Model conversion validated successfully.")



def convert_and_write_model(input_dir: str, output_dir: str):
    """Convert the model and save it (this implicitly save the config as well)."""

    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    config = DogeConfig.from_pretrained(input_dir)

    old_model = OldDogeForCausalLM.from_pretrained(input_dir)
    full_state_dict = {}
    original_state_dict = load_file(f"{input_dir}/model.safetensors")
    new_state_dict = convert_state_dict(original_state_dict, config)
    full_state_dict.update(new_state_dict)

    model = DogeForCausalLM(config)
    model.load_state_dict(full_state_dict, strict=False, assign=True)
    # tie_word_embeddings
    model.lm_head.weight = model.model.word_embed.weight

    validate_converted_model(old_model, model, tokenizer)
    model.save_pretrained(output_dir)




if __name__ == "__main__":
   
    convert_and_write_model(
        input_dir="./Doge-160M",
        output_dir="./Doge-160M-new",
    )