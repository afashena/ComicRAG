from pydantic import BaseModel
from peft import LoraConfig
from trl import SFTConfig
import torch

class Config(BaseModel):
    model_path: str
    train_dir: str
    lora_config: dict
    sft_config: dict

    def build_lora(self) -> LoraConfig:
        return LoraConfig(**self.lora_config)
    
    def build_sft(self) -> SFTConfig:
        cfg = self.sft_config.copy()
        if cfg.get("bf16") is None and cfg.get("fp16") is None:
            cfg["bf16"] =  torch.cuda.is_bf16_supported()
            cfg["fp16"] = not torch.cuda.is_bf16_supported()
        return SFTConfig(**cfg)