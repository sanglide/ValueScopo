#!/usr/bin/env python3
"""
call_valueguard_api.py

Usage:
  python call_valueguard_api.py --chunks chunks.json --out per_chunk_results.json \
    --provider deepseek --model ds-1 --temperature 0.0 --max-tokens 900 --concurrency 4

linux:
export DEEPSEEK_API_KEY=sk-xxxx

win:
$env:DEEPSEEK_API_KEY = sk-xxxx

python ValueHypothesis/call_valueguard_api.py --chunks results/tensorflow_tensorflow_pr_chunks.json --out results/tensorflow_tensorflow_per_chunk_results_deepseek.json --provider deepseek --model deepseek-chat --temperature 0.0 --max-tokens 900 --concurrency 4

This script:
  - Loads chunk objects from a JSON file (array of chunk dicts).
  - For each chunk, builds a prompt (from prompts/per_chunk_prompt.txt) and calls the chosen LLM provider.
  - Parses provider response into JSON, validates against per-chunk schema (schemas/per_chunk_schema.json).
  - Writes an array of per-chunk outputs to the output file.
  - Supports multiple providers via a small provider interface.

Providers:
  - openai: example for OpenAI-compatible APIs (openai.com)
  - deepseek: placeholder adapter (fill API details according to DeepSeek's documentation)
  - qianwen: placeholder adapter for 千问 (fill per-provider details)

Dependencies:
  pip install requests jsonschema tqdm
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import requests
from jsonschema import validate, ValidationError
from tqdm import tqdm

# -------------------------
# Config / Defaults
# -------------------------
DEFAULT_PROMPT_PATH = f"Prompts/per_chunk_prompt.txt"
PER_CHUNK_SCHEMA_PATH = f"Schemas/per_chunk_schema.json"

# -------------------------
# Utilities
# -------------------------


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_json_load_from_text(text: str) -> Optional[Any]:
    """
    Try to parse text as JSON. If fails, attempt a simple repair:
      - find first '{' and last '}' and parse substring
    Return None if not parseable.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.strip()
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and j > i:
            candidate = text[i : j + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None


# -------------------------
# Provider Interface
# -------------------------


class LLMProvider:
    """
    Base class for provider adapters. Subclass must implement send_request().
    """

    def __init__(self, model: str, temperature: float, max_tokens: int, timeout: int = 60):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        """
        Send messages to the model and return text (string).
        messages is a list of {"role": "...", "content": "..."} dicts.
        """
        raise NotImplementedError("send_request must be implemented by subclass.")


class OpenAIProvider(LLMProvider):
    """
    Example OpenAI-compatible provider adapter.
    Uses environment variable OPENAI_API_KEY.
    """

    def __init__(self, model: str, temperature: float, max_tokens: int, timeout: int = 60):
        super(OpenAIProvider, self).__init__(model, temperature, max_tokens, timeout)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIProvider")

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(self.api_url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # Typical OpenAI response path:
        return data["choices"][0]["message"]["content"]


class DeepSeekProvider(LLMProvider):
    """
    Template adapter for `deepseek` provider.

    NOTE: Adapt the `url`, request body, and response parsing to DeepSeek's actual API.
    Set environment:
      - DEEPSEEK_API_KEY
      - DEEPSEEK_API_URL (optional; provide actual endpoint)
    """

    def __init__(self, model: str, temperature: float, max_tokens: int, timeout: int = 60):
        super(DeepSeekProvider, self).__init__(model, temperature, max_tokens, timeout)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeekProvider")

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        # Example body format: adjust according to provider spec
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(self.api_url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # EXTRACT TEXT: adapt to the real response structure
        # e.g., data.get('result', {}).get('content', '')
        # Example placeholder:
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("message", {}).get("content", "")
        # Fallbacks (common alternative field names)
        if "result" in data and isinstance(data["result"], dict):
            return data["result"].get("output_text", "") or data["result"].get("text", "")
        # As last resort, stringify
        return json.dumps(data)


class QianwenProvider(LLMProvider):
    """
    Qianwen (DashScope) OpenAI-compatible adapter.
    """

    def __init__(self, model: str, temperature: float, max_tokens: int, timeout: int = 60):
        super(QianwenProvider, self).__init__(model, temperature, max_tokens, timeout)
        self.api_key = os.getenv("QIANWEN_API_KEY")
        self.api_url = os.getenv(
            "QIANWEN_API_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        )
        if not self.api_key:
            raise RuntimeError("QIANWEN_API_KEY is required")

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        body = {
            "model": self.model,           # e.g. qwen-plus
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": "Bearer %s" % self.api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(self.api_url, json=body, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]



# -------------------------
# Orchestration
# -------------------------


def build_messages_from_prompt(prompt_template: str, chunk: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Replace a placeholder {{INPUT_JSON}} in prompt_template with the chunk JSON.
    Returns a message list for chat-style APIs.
    """
    input_json_str = json.dumps(chunk, ensure_ascii=False)
    user_content = prompt_template.replace("{{INPUT_JSON}}", input_json_str)
    return [{"role": "system", "content": "You are ValueGuard Assistant. Return only JSON that follows the schema."},
            {"role": "user", "content": user_content}]


def get_provider(provider_name: str, model: str, temperature: float, max_tokens: int, timeout: int) -> LLMProvider:
    name = provider_name.lower()
    if name == "openai":
        return OpenAIProvider(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if name == "deepseek":
        return DeepSeekProvider(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if name in ("qianwen", "千问", "qianwen-provider"):
        return QianwenProvider(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    raise ValueError("Unknown provider: %s" % provider_name)


def validate_per_chunk_output(obj: dict, schema: dict) -> Tuple[bool, Optional[str]]:
    try:
        validate(instance=obj, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e)


def call_chunk(provider: LLMProvider, prompt_template: str, chunk: Dict[str, Any], schema: dict,
               timeout: int, retry: int) -> Dict[str, Any]:
    """
    Call provider for a single chunk. Returns a per-chunk result dict (schema-compliant if possible).
    """
    messages = build_messages_from_prompt(prompt_template, chunk)
    last_err = None
    for attempt in range(max(1, retry + 1)):
        try:
            text = provider.send_request(messages)
            # try parse
            parsed = safe_json_load_from_text(text)
            if parsed is None:
                # attempt to salvage by looking for top-level object in multiple lines (already done) or fallback to minimal result
                result = {
                    "chunk_id": chunk.get("chunk_id"),
                    "findings": [],
                    "summary": "invalid_json_from_model",
                }
                return result
            # validate schema
            ok, err = validate_per_chunk_output(parsed, schema)
            if ok:
                return parsed
            else:
                # if schema validation fails, return partial with note
                return {
                    "chunk_id": parsed.get("chunk_id", chunk.get("chunk_id")),
                    "findings": parsed.get("findings", parsed.get("findings", [])),
                    "summary": "schema_validation_failed: %s" % err
                }
        except Exception as e:
            last_err = e
            # simple backoff
            time.sleep(1 + attempt * 2)
    # on total failure
    return {
        "chunk_id": chunk.get("chunk_id"),
        "findings": [],
        "summary": "error_calling_llm: %r" % (last_err,)
    }


def process_all_chunks(chunks: List[Dict[str, Any]], provider_name: str, model: str,
                       prompt_template: str, schema: dict, concurrency: int, temperature: float,
                       max_tokens: int, timeout: int, retry: int) -> List[Dict[str, Any]]:
    provider = get_provider(provider_name, model, temperature, max_tokens, timeout)
    results: List[Dict[str, Any]] = []
    # Use ThreadPoolExecutor to parallelize calls but keep a conservative concurrency.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as exe:
        futures = {exe.submit(call_chunk, provider, prompt_template, chunk, schema, timeout, retry): chunk for chunk in chunks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM calls"):
            try:
                res = fut.result()
            except Exception as e:
                chunk = futures[fut]
                res = {"chunk_id": chunk.get("chunk_id"), "findings": [], "summary": "executor_error: %r" % (e,)}
            results.append(res)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="chunks.json", help="Input chunks.json (array of chunk objects)")
    parser.add_argument("--out", default="per_chunk_results.json", help="Output file path")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"), help="LLM provider name")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"), help="Model/engine name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for model")
    parser.add_argument("--max-tokens", type=int, default=900, help="Max tokens for model response")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel requests")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout (s)")
    parser.add_argument("--retry", type=int, default=2, help="Retry attempts on failure")
    args = parser.parse_args()

    # load prompt
    prompt_path = os.getenv("PER_CHUNK_PROMPT_PATH", DEFAULT_PROMPT_PATH)
    if not os.path.exists(prompt_path):
        print("Prompt file not found at %s" % prompt_path, file=sys.stderr)
        sys.exit(1)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # load chunks
    if not os.path.exists(args.chunks):
        print("Chunks file not found: %s" % args.chunks, file=sys.stderr)
        sys.exit(1)
    chunks = load_json(args.chunks)
    if not isinstance(chunks, list):
        print("Chunks file must contain a JSON array of chunk objects", file=sys.stderr)
        sys.exit(1)

    # load schema
    if not os.path.exists(PER_CHUNK_SCHEMA_PATH):
        print("Schema file not found at %s" % PER_CHUNK_SCHEMA_PATH, file=sys.stderr)
        sys.exit(1)
    schema = load_json(PER_CHUNK_SCHEMA_PATH)

    # orchestrate
    results = process_all_chunks(
        chunks=chunks,
        provider_name=args.provider,
        model=args.model,
        prompt_template=prompt_template,
        schema=schema,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retry=args.retry,
    )

    write_json(results, args.out)
    print("Wrote %d per-chunk results to %s" % (len(results), args.out))


if __name__ == "__main__":
    main()
