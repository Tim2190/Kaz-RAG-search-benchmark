"""
Бэкенды LLM-генерации для RAG. Интерфейс простой: generate(prompt) -> str.

- HFTransformersLLM — открытые instruct-модели локально/в Colab:
  IBM Granite (ibm-granite/granite-3.1-2b-instruct) и
  Google Gemma (google/gemma-2-2b-it). Бесплатно, имена IBM/Google — честно.
- EchoLLM — заглушка для тестов (без сети/моделей).

Тяжёлые модели грузятся лениво и работают в Colab (GPU желателен).
"""

from __future__ import annotations

from typing import Optional, Protocol

# Пресеты instruct-моделей: короткое имя -> HF id
LLM_MODELS = {
    "granite":   "ibm-granite/granite-3.1-2b-instruct",  # IBM, 2B (слабоват в kk)
    "granite8b": "ibm-granite/granite-3.1-8b-instruct",  # IBM, 8B
    "gemma":     "google/gemma-2-2b-it",                  # Google, 2B
    "gemma9b":   "google/gemma-2-9b-it",                  # Google, 9B (gated)
    "qwen7b":    "Qwen/Qwen2.5-7B-Instruct",              # Alibaba, силён в kk
}


class LLM(Protocol):
    def generate(self, prompt: str) -> str:  # pragma: no cover - интерфейс
        ...


class HFTransformersLLM:
    """Чат-модель из transformers. Грузится лениво."""

    def __init__(self, model_key: str, max_new_tokens: int = 64,
                 device: Optional[str] = None, load_in_4bit: bool = False) -> None:
        self.model_id = LLM_MODELS.get(model_key, model_key)
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is None:
            import torch
            import transformers
            from transformers import pipeline
            transformers.logging.set_verbosity_error()  # убрать спам предупреждений
            model_kwargs = {"torch_dtype": torch.bfloat16}
            if self.load_in_4bit:
                # 4-битная загрузка — чтобы 7-9B модели влезли в бесплатный T4 (16 ГБ)
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self._pipe = pipeline(
                "text-generation", model=self.model_id,
                device_map="auto", model_kwargs=model_kwargs,
            )
            # снять зашитый max_length=20, чтобы не конфликтовал с max_new_tokens
            try:
                self._pipe.model.generation_config.max_length = None
                if self._pipe.tokenizer.pad_token_id is None:
                    self._pipe.tokenizer.pad_token_id = self._pipe.tokenizer.eos_token_id
            except Exception:
                pass
        return self._pipe

    def generate(self, prompt: str) -> str:
        pipe = self._get_pipe()
        messages = [{"role": "user", "content": prompt}]
        out = pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        # transformers возвращает сгенерированный диалог; берём последний ответ
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"].strip()
        return str(generated).strip()


class EchoLLM:
    """Тестовая заглушка: возвращает заранее заданный ответ на любой промпт
    или конкретный ответ по подстроке вопроса (для юнит-тестов)."""

    def __init__(self, default: str = "Ақпарат жоқ", rules: Optional[dict] = None) -> None:
        self.default = default
        self.rules = rules or {}

    def generate(self, prompt: str) -> str:
        for needle, answer in self.rules.items():
            if needle in prompt:
                return answer
        return self.default
