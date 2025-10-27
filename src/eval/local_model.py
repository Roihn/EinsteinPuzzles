
import torch

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class LoRAChatLLaMAModel:
    def __init__(self, 
                 base_model_name: str, 
                 lora_path: str, 
                 device: str = "cuda"):
        """
        Initialize the model with a base Hugging Face model and a LoRA adapter.
        
        Args:
            base_model_name (str): The Hugging Face model name.
            lora_path (str): Path to the fine-tuned LoRA adapter.
        """
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        # Load base model - removing device_map="auto" to respect specified device
        if self.device == "cuda":
            base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.float16
            )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, lora_path)

        # Move model to GPU if available
        self.model.to(self.device)
    
    def predict(self, messages, max_tokens=64, temperature=0.2):
        """
        Generate a response from the chat model using LoRA.
        
        Args:
            messages (list): List of message dictionaries with "role" and "content" keys.
            max_tokens (int): Maximum number of tokens for generation.
            temperature (float): Sampling temperature.
        
        Returns:
            str: The generated response.
        """
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors='pt', padding="longest", max_length=2048, add_generation_prompt=True).to(self.device)
        
        # Create attention mask (all 1s since we're padding to longest)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # Generate response
        with torch.no_grad():
            # Create a dictionary of generation parameters
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_length": input_ids.shape[1] + max_tokens,
                "temperature": temperature,
                "eos_token_id": self.eot_token_id
            }
            
            output = self.model.generate(**generation_kwargs)
        # Decode response
        response_text = self.tokenizer.decode(output[:, input_ids.shape[1]:][0], skip_special_tokens=False)
        return response_text
    
    def predict_n(
        self,
        messages: List[dict],
        n: int = 4,
        max_tokens: int = 192,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate `n` independent chat completions (no best-of filtering).

        Args:
            messages:  Same structure you send to vLLM.
            n:         Number of samples to return (= payload["n"] in vLLM call).
            max_tokens:Maximum new tokens per sample (= payload["max_tokens"]).
            temperature, top_p:  Same sampling knobs you pass to vLLM.

        Returns:
            List[str] of length `n`, each a decoded response string.
        """
        # 1. Encode the prompt once
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding="longest",
            max_length=4500,   # keep your original safety cap
        ).to(self.device)

        attention_mask = torch.ones_like(input_ids).to(self.device)

        # 2. Sampling generation
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_tokens,
            do_sample=True,                 # enable stochastic decoding
            num_return_sequences=n,         # = vLLM payload["n"]
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.eot_token_id, # stop at <|eot_id|>
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            sequences = self.model.generate(**gen_kwargs)

        # 3. Decode and strip the stop token if it appears
        start = input_ids.shape[1]         # position where new text begins
        raw_responses = [
            self.tokenizer.decode(seq[start:], skip_special_tokens=True)
            for seq in sequences
        ]

        # Remove everything after the first explicit <|eot_id|> (if any)
        clean_responses = [
            resp.split("<|eot_id|>")[0].rstrip() for resp in raw_responses
        ]

        return clean_responses

class LoRAChatQwenModel:
    """
    Qwen-2.5-7B-Instruct + LoRA adapter, compatible with the same
    `messages = [{"role":"system","content":...}, ...]` format you
    passed to Llama-3.
    """
    def __init__(
        self,
        base_model_name: str,
        lora_path: str,
        device: str = "cuda"
    ):
        self.device = device
        print('using qwen model')
        # --- Tokenizer & special-token setup ---------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True        # required for Qwen
        )
        # Qwen uses <|endoftext|> (151 643) as EOS/BOS/PAD
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stop_ids = [self.tokenizer.eos_token_id]          # [151643]
        # If you want to be extra safe you can also stop on <|im_end|>
        try:
            self.stop_ids.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        except KeyError:
            pass                                               # base models lack it

        # --- Base model - removing device_map="auto" ------------------------------------------------------
        if self.device == "cuda":
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        # --- LoRA adapter ----------------------------------------------------
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.to(self.device)
        self.model.eval()

    # -------------------------------------------------------------------------
    def _encode(self, messages, max_ctx=4500):
        return self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding="longest",
            max_length=max_ctx,
            add_generation_prompt=True,
        ).to(self.device)

    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def predict(self, messages, max_tokens=192, temperature=0.2):
        input_ids = self._encode(messages)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_length=input_ids.shape[1] + max_tokens,
            temperature=temperature,
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = outputs[:, input_ids.shape[1]:][0]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def predict_n(
        self,
        messages: List[dict],
        n: int = 4,
        max_tokens: int = 192,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> List[str]:
        input_ids = self._encode(messages)
        sequences = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_length=input_ids.shape[1] + max_tokens,
            do_sample=True,
            num_return_sequences=n,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        start = input_ids.shape[1]
        texts = [
            self.tokenizer.decode(seq[start:], skip_special_tokens=True).strip()
            for seq in sequences
        ]
        return texts