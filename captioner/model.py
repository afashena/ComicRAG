import json
import re
from PIL import Image
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
import torch

from config import Config


class QwenCaptioner:
    def __init__(self, config: Config, system_prompt: str, user_prompt:str, schema: type[BaseModel] | None = None, use_4bit: bool = True):
        self.config = config
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.schema = schema
        self.use_4bit = use_4bit

    def load_qwen(
            self,
            ) -> tuple:
        """
        Load Qwen2.5-VL model and processor.

        Args:
            model_path: Path to the local model directory.
            use_4bit: quantize to 4-bit for low VRAM GPUs (your GTX 1050).
                    Set False on Lambda Cloud where VRAM is not a constraint.

        Returns:
            (model, processor) tuple — pass both to caption_image().
        """
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            # Cap image token budget to keep VRAM manageable on small GPUs.
            # 256*28*28 = ~200k pixels minimum, 512*28*28 = ~400k pixels maximum.
            # Raise max_pixels on Lambda if you want more visual detail.
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                #llm_int8_enable_fp32_cpu_offload=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                quantization_config=bnb_config,
                device_map="cuda:0",
                low_cpu_mem_usage=True,
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
                attn_implementation="sdpa",
            )

        #self.model.eval()

    def load_finetuned_model(self, adapter_path: str):
        """
        Load the base model and apply the saved LoRA adapter for inference.
        Use this after training to test your fine-tuned model.
        """

        self.processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)

        # Use 4-bit quantization to match baseline model and save memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_path,
            quantization_config=bnb_config,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Merge LoRA adapters into the base model for faster inference
        self.model = self.model.merge_and_unload()
        
        self.model.eval()


    def enforce_no_additional_properties(self, schema: dict) -> dict:
        """
        Recursively enforce additionalProperties=false
        on all object schemas (OpenAI strict requirement)
        """
        if not isinstance(schema, dict):
            return schema

        schema_type = schema.get("type")

        if schema_type == "object":
            schema["additionalProperties"] = False

            for prop in schema.get("properties", {}).values():
                schema = self.enforce_no_additional_properties(prop)

            # Required for OpenAI: explicitly define required
            if "required" not in schema and "properties" in schema:
                schema["required"] = list(schema["properties"].keys())

        elif schema_type == "array":
            schema = self.enforce_no_additional_properties(schema.get("items"))

        # Handle anyOf / oneOf / allOf (rare but safe)
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                for subschema in schema[key]:
                    schema = self.enforce_no_additional_properties(subschema)

        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                schema = self.enforce_no_additional_properties(def_schema)

        return schema


# ── 2. Core inference function ────────────────────────────────────────────────

    def caption_image(
            self,
            image_path: str,
            max_new_tokens: int = 1024,
        ) -> BaseModel | str:
        """
        Run Qwen2.5-VL on a single image and return either a Pydantic object
        (if schema is provided) or a raw string.

        Args:
            image_path:     Path to the image file.
            model:          Loaded Qwen model from load_qwen().
            processor:      Loaded processor from load_qwen().
            system_prompt:  Instruction context — who the model is and what it should do.
            user_prompt:    The specific question or instruction for this image.
            schema:         Optional Pydantic model class. If provided, the system prompt
                            is extended with the JSON schema and the output is parsed and
                            validated. If None, returns the raw string response.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Validated Pydantic instance if schema is provided, otherwise a plain string.
        """
        image = Image.open(image_path).convert("RGB")

        # If a schema is provided, extend the system prompt with the schema
        # and instruct the model to return only JSON.
        # This mirrors the approach that worked for you with Ollama.
        if self.schema is not None:
            schema_parsed = self.enforce_no_additional_properties(self.schema.model_json_schema())
            schema_str = json.dumps(schema_parsed, indent=2)
            full_system_prompt = (
                f"{self.system_prompt}\n\n"
                f"Always respond with a valid JSON object matching this schema exactly:\n"
                f"{schema_str}\n"
                f"Return only the JSON object. No markdown, no explanation, no code fences."
            )
        else:
            full_system_prompt = self.system_prompt

        messages = [
            {
                "role": "system",
                "content": full_system_prompt,
            },
            {
                "role": "user",
                "content": [
                    # Qwen requires the image dict to use "image" key with a PIL Image
                    # or a file path string — both work via process_vision_info
                    {"type": "image", "image": image},
                    {"type": "text",  "text": self.user_prompt},
                ],
            },
        ]

        # apply_chat_template produces the formatted text string
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # process_vision_info extracts image tensors from the message structure
        # This is Qwen-specific and required — do not skip it
        image_inputs, video_inputs = process_vision_info(messages)

        # Wrap entire inference in torch.inference_mode for maximum speed
        # This disables gradient computation and enables optimizations
        with torch.inference_mode():
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # greedy — deterministic, better for structured output
            )

            # Trim the prompt tokens from the output — decode only what was generated
            trimmed = generated_ids[0][input_len:]
            raw_output = self.processor.decode(trimmed, skip_special_tokens=True).strip()

        if self.schema is None:
            return raw_output

        return self._parse_to_schema(raw_output)


    def _parse_to_schema(self, raw: str) -> BaseModel:
        """
        Extract JSON from model output and validate against the Pydantic schema.
        Handles markdown fences and leading/trailing text gracefully.
        """
        # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        else:
            # Find the outermost {...} block
            brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
            candidate = brace_match.group(0) if brace_match else raw

        try:
            data = json.loads(candidate)
            return self.schema.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Could not parse model output as {self.schema.__name__}.\n"
                f"Raw output was:\n{raw}\n"
                f"Error: {e}"
            )
        
    def get_structured_prompt(self, image: Image, gt_label: str | None = None) -> str:
        message = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": self.user_prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        # The ground truth label — what the model learns to produce
                        "content": gt_label,
                    },
                ]
        
        if gt_label is None:
            message.pop()  # Remove assistant part for inference prompt
            return message
        return message
