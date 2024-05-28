import torch
import json
import os
model_name = "wanda-30b-2:4-spp"
dir = "/path/to/output/{}/{}/".format(model_name, model_name)
os.makedirs(dir, exist_ok=True)

adapter_config = {
    "base_model_name_or_path": "",
    "bias": "none",
    "enable_lora": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "merge_weights": False,
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": 8,
    "target_modules": [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "score"
    ],
    "task_type": "CAUSAL_LM"
}
json.dump(adapter_config, open("{}/adapter_config.json".format(dir), "w"))
ckpt_full = torch.load("/path/to/output/{}/checkpoint-final/pytorch_model.bin".format(model_name), map_location="cpu")
ckpt = {}

for key in ckpt_full.keys():
    if "lora" in key:
        ckpt[key] = ckpt_full[key]

torch.save(ckpt, "{}/adapter_model.bin".format(dir))


