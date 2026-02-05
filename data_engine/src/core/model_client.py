"""
Model client for interacting with various LLM backends.

Supports:
- vLLM server (recommended for production)
- Hugging Face transformers
- OpenAI API
- Anthropic API
"""

import copy
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# For reproducibility
SEED = 1234
set_seed(SEED)
random.seed(42)


class LMClient:
    """
    A unified client for interacting with various language model backends.
    
    This client provides a consistent interface for generating text from different
    model backends including vLLM servers, Hugging Face models, OpenAI, and Anthropic.
    
    Args:
        model_name: The name or path of the model to use.
        model_type: The type of backend ("vllm_server", "vllm", "hf", "openai", "anthropic").
        model_url: Optional URL for the model server.
        num_rollouts: Number of sequences to generate per request.
        tokenizer: Optional pre-loaded tokenizer.
        **model_args: Additional model-specific arguments including:
            - max_tokens: Maximum tokens to generate.
            - temperature_range: Tuple of (min, max) temperature values.
            - top_p: Nucleus sampling parameter.
            - port: Server port for vllm_server mode.
    
    Example:
        >>> client = LMClient(
        ...     model_name="Qwen/Qwen2-VL-7B-Instruct",
        ...     model_type="vllm_server",
        ...     port=8000,
        ...     max_tokens=4096,
        ...     temperature_range=(0.6, 0.9),
        ...     top_p=0.95
        ... )
        >>> response, temp = client.generate("Hello, world!")
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
        model_type: str = "vllm_server",
        model_url: str = "",
        num_rollouts: int = 5,
        tokenizer: Optional[Any] = None,
        **model_args
    ):
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.num_rollouts = num_rollouts
        self.tokenizer = tokenizer
        
        # Model generation parameters
        self.max_tokens = model_args.get('max_tokens', 4096)
        self.temperature_range = model_args.get('temperature_range', (0.6, 0.9))
        self.top_p = model_args.get('top_p', 0.95)
        self.port = model_args.get("port", 8000)
        
        # Set up server URL
        self.url = model_url or f"http://127.0.0.1:{self.port}/v1/completions"
        print(f"[LMClient] Initialized with model: {self.model_name}")
        print(f"[LMClient] Backend: {self.model_type}, URL: {self.url}")
        
        # Base payload for API requests
        self.payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n": self.num_rollouts
        }
        
        self.__dict__.update(model_args)
        
        # Initialize backend-specific components
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the appropriate backend based on model_type."""
        if self.model_type == "vllm":
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=self.model_name, enable_prefix_caching=True)
            self.SamplingParams = SamplingParams
        elif self.model_type == "hf":
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16, 
                device_map="cuda"
            )
        elif self.model_type == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model_type == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.model_type == "vllm_server":
            pass  # No additional initialization needed
        else:
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                "Choose from 'vllm', 'vllm_server', 'hf', 'openai', or 'anthropic'."
            )
    
    def generate(
        self,
        prompt: str,
        num_rollouts: Optional[int] = None,
        is_greedy: bool = False,
        **kwargs
    ) -> Tuple[List[str], float]:
        """
        Generate text from the model.
        
        Args:
            prompt: The input prompt.
            num_rollouts: Number of sequences to generate (overrides default).
            is_greedy: Whether to use greedy decoding.
            **kwargs: Additional generation parameters.
        
        Returns:
            A tuple of (generated_texts, temperature_used).
        """
        if num_rollouts is None:
            num_rollouts = self.num_rollouts
            
        if self.model_type == 'vllm_server':
            return self.generate_vllm_server(prompt, num_rollouts, is_greedy, **kwargs)
        elif self.model_type == "vllm":
            return self.generate_vllm(prompt, num_rollouts)
        elif self.model_type == "hf":
            return self.generate_hf(prompt, num_rollouts)
        elif self.model_type in ("anthropic", "openai"):
            return self.generate_api(prompt, num_rollouts)
        
        raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def generate_vl_vllm_server(
        self,
        messages: List[Dict],
        num_rollouts: Optional[int] = None,
        is_greedy: bool = False,
        **kwargs
    ) -> Tuple[Tuple[List[str], List, List], float]:
        """
        Generate text from a vision-language model via vLLM server.
        
        Args:
            messages: List of message dictionaries with role and content.
            num_rollouts: Number of sequences to generate.
            is_greedy: Whether to use greedy decoding.
            **kwargs: Additional parameters.
        
        Returns:
            A tuple of ((texts, logprobs, offsets), temperature).
        """
        if num_rollouts is None:
            num_rollouts = self.num_rollouts
            
        temperature = np.random.uniform(
            low=self.temperature_range[0],
            high=self.temperature_range[1]
        )
        
        # Use chat completions endpoint for VL models
        chat_url = self.url.replace("/v1/completions", "/v1/chat/completions")
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature if not is_greedy else 0,
            "top_p": self.top_p if not is_greedy else 1,
            "n": num_rollouts if not is_greedy else 1,
        }
        
        if is_greedy:
            payload['temperature'] = 0
            payload['n'] = 1
        
        try:
            response = requests.post(
                chat_url,
                headers={"User-Agent": "CogDoc-DataEngine"},
                json=payload,
                stream=False,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                texts = [choice['message']['content'] for choice in result['choices']]
                # Note: logprobs might not be available in all responses
                logprobs = [None] * len(texts)
                offsets = [None] * len(texts)
                return (texts, logprobs, offsets), temperature
            else:
                print(f"[LMClient] Error: {response.status_code}, {response.text}")
                return ([], [], []), temperature
                
        except Exception as e:
            print(f"[LMClient] Request failed: {e}")
            return ([], [], []), temperature
    
    def generate_vllm_server(
        self,
        prompt: str,
        num_rollouts: Optional[int] = None,
        is_greedy: bool = False,
        special_stop: Optional[List[str]] = None,
        prompt_only: bool = False
    ) -> Tuple[Tuple[List[str], List, List], float]:
        """
        Generate text using vLLM server backend.
        
        Args:
            prompt: The input prompt string.
            num_rollouts: Number of sequences to generate.
            is_greedy: Whether to use greedy decoding.
            special_stop: Optional stop sequences.
            prompt_only: If True, return log probabilities for the prompt only.
        
        Returns:
            A tuple of ((texts, logprobs, offsets), temperature).
        """
        temperature = np.random.uniform(
            low=self.temperature_range[0],
            high=self.temperature_range[1]
        )
        
        payload = copy.copy(self.payload)
        payload.update({
            "temperature": temperature,
            "prompt": prompt,
            "logprobs": 1 if not is_greedy else 0,
        })
        
        if num_rollouts is not None:
            payload['n'] = num_rollouts
            
        if is_greedy:
            payload['top_k'] = 1
            payload['top_p'] = 1
            payload['temperature'] = 0
            payload['n'] = 1
        
        if special_stop:
            payload['stop'] = special_stop
        
        if prompt_only:
            payload.update({
                'prompt': prompt,
                'n': 1,
                'temperature': 1.0,
                'prompt_logprobs': 1
            })
        
        try:
            response = requests.post(
                self.url,
                headers={"User-Agent": "CogDoc-DataEngine"},
                json=payload,
                stream=False,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                texts = [item['text'] for item in result['choices']]
                
                if prompt_only:
                    logprobs = [item['prompt_logprobs'] for item in result['choices']]
                else:
                    logprobs = [
                        item['logprobs']['token_logprobs'] 
                        for item in result['choices']
                    ]
                
                offsets = result
                return (texts, logprobs, offsets), temperature
            else:
                print(f"[LMClient] Error: {response.status_code}, {response.text}")
                return ([], [], []), temperature
                
        except Exception as e:
            print(f"[LMClient] Request failed: {e}")
            return ([], [], []), temperature
    
    def generate_vllm(
        self,
        prompt: str,
        num_rollouts: int
    ) -> Tuple[List[str], float]:
        """Generate using local vLLM."""
        temperature = random.choice(self.temperature_range)
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=num_rollouts,
            seed=SEED,
        )
        
        start_time = time.time()
        outputs = self.llm.generate(prompt, sampling_params)
        elapsed = time.time() - start_time
        print(f'[LMClient] Generated {num_rollouts} responses in {elapsed:.2f}s')
        
        results = [
            completion.text 
            for output in outputs 
            for completion in output.outputs
        ]
        return results, temperature
    
    def generate_hf(
        self,
        prompt: str,
        num_rollouts: int
    ) -> Tuple[List[str], float]:
        """Generate using Hugging Face transformers."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        results = []
        
        for _ in range(num_rollouts):
            temperature = random.uniform(
                self.temperature_range[0],
                self.temperature_range[1]
            )
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=self.max_tokens,
                temperature=temperature,
                num_return_sequences=1
            )
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(result)
        
        return results, temperature
    
    def generate_api(
        self,
        prompt: str,
        num_rollouts: int
    ) -> Tuple[List[str], float]:
        """Generate using OpenAI or Anthropic API."""
        raise NotImplementedError(
            "API-based generation is not fully implemented. "
            "Please use vllm_server mode for production."
        )
