#!/usr/bin/env python3
"""
aggregate_results.py

Orchestrates:
  1) (optionally) generate chunks from a remote public repo by calling generate_chunks.py
  2) call per-chunk LLM analysis by invoking call_valueguard_api.py
  3) aggregate per-chunk outputs into final_report.json using aggregator prompt + LLM

Usage examples:

# 1) operate on local chunks.json (already generated)
python aggregate_results.py --chunks chunks.json --per-chunk-out per_chunk_results.json --final-out final_report.json \
  --provider openai --model gpt-4o-mini

# 2) specify a remote public repo and limit to 50 chunks
python aggregate_results.py --repo octocat/Hello-World --num-chunks 50 \
  --provider deepseek --model deepseek-chat --concurrency 2

Notes:
- This script expects generate_chunks.py and call_valueguard_api.py to be present and executable with the same Python interpreter.
- Place your provider API keys in environment variables:
  OPENAI_API_KEY, DEEPSEEK_API_KEY, QIANWEN_API_KEY as appropriate.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests
from jsonschema import validate, ValidationError

# Paths (assumed relative; change if your layout differs)
GENERATE_SCRIPT = "../ChangeSense/generate_chunks.py"
CALL_SCRIPT = "../ValueHypothesis/call_valueguard_api.py"

PER_CHUNK_SCHEMA_PATH = f"../Schemas/per_chunk_schema.json"
AGG_SCHEMA_PATH = f"../Schemas/aggregator_schema.json"
PER_CHUNK_PROMPT_PATH = f"../Prompts/per_chunk_prompt.txt"

# -------------------------
# Simple OpenAI-compatible provider adapters (for aggregator call)
# -------------------------


class LLMProvider:
    def __init__(self, model: str, timeout: int = 60):
        self.model = model
        self.timeout = timeout

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError()


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, timeout: int = 60):
        super(OpenAIProvider, self).__init__(model, timeout)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIProvider")

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": 2000}
        headers = {"Authorization": "Bearer %s" % self.api_key, "Content-Type": "application/json"}
        resp = requests.post(self.api_url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class DeepSeekProvider(LLMProvider):
    def __init__(self, model: str, timeout: int = 60):
        super(DeepSeekProvider, self).__init__(model, timeout)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeekProvider")

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": 2000}
        headers = {"Authorization": "Bearer %s" % self.api_key, "Content-Type": "application/json"}
        resp = requests.post(self.api_url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class QianwenProvider(LLMProvider):
    def __init__(self, model: str, timeout: int = 60):
        super(QianwenProvider, self).__init__(model, timeout)
        self.api_key = os.getenv("QIANWEN_API_KEY")
        self.api_url = os.getenv("QIANWEN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        if not self.api_key:
            raise RuntimeError("QIANWEN_API_KEY is required for QianwenProvider")

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "max_tokens": 2000}
        headers = {"Authorization": "Bearer %s" % self.api_key, "Content-Type": "application/json"}
        resp = requests.post(self.api_url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def get_provider(provider_name: str, model: str, timeout: int = 60) -> LLMProvider:
    name = provider_name.lower()
    if name == "openai":
        return OpenAIProvider(model=model, timeout=timeout)
    if name == "deepseek":
        return DeepSeekProvider(model=model, timeout=timeout)
    if name in ("qianwen", "千问", "qwen"):
        return QianwenProvider(model=model, timeout=timeout)
    raise ValueError("Unknown provider: %s" % provider_name)


# -------------------------
# Helpers
# -------------------------


def run_subprocess(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    """Run external script synchronously and raise on error, streaming output."""
    print("Running:", " ".join(cmd))
    completed = subprocess.run([sys.executable] + cmd, env=env or os.environ, check=False)
    if completed.returncode != 0:
        raise RuntimeError("Command failed: %s (exit %d)" % (" ".join(cmd), completed.returncode))


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        text = text.strip()
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except Exception:
                return None
        return None


# -------------------------
# Aggregation logic
# -------------------------


def build_aggregator_input(chunks_results: List[Dict[str, Any]], repo: str, pr_id: str) -> Dict[str, Any]:
    # Build a compact repo metadata; README summary can be empty for MVP
    repo_metadata = {"README_summary": "", "changed_files_count": len(chunks_results)}
    return {"repo": repo, "pr_id": pr_id, "chunks": chunks_results, "repo_metadata": repo_metadata}


def call_aggregator_llm(provider_name: str, model: str, prompt_template: str, input_obj: Dict[str, Any],
                        timeout: int = 60) -> Dict[str, Any]:
    provider = get_provider(provider_name, model, timeout)
    # messages: system + user
    # system: instruct aggregator to return only JSON (prompt template should already require that)
    messages = [
        {"role": "system", "content": "You are ValueGuard Aggregator. Return only JSON matching schema."},
        {"role": "user", "content": prompt_template.replace("{{INPUT_JSON}}", json.dumps(input_obj, ensure_ascii=False))}
    ]
    text = provider.send_request(messages)
    parsed = safe_parse_json_from_text(text)
    if parsed is None:
        raise RuntimeError("Aggregator LLM returned non-JSON or unparsable text.")
    return parsed


def validate_aggregator_output(obj: Dict[str, Any], schema_path: str) -> None:
    schema = load_json(schema_path)
    try:
        validate(instance=obj, schema=schema)
    except ValidationError as e:
        raise RuntimeError("Aggregator output did not validate against schema: %s" % str(e))


# -------------------------
# Main orchestration
# -------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", help="Optional remote public repo in owner/repo format to clone and analyze")
    p.add_argument("--depth", type=int, default=2, help="Shallow clone depth when cloning remote repo (default 2)")
    p.add_argument("--num-chunks", type=int, default=200, help="Maximum number of chunks to analyze (default 200)")
    p.add_argument("--chunks-out", default="chunks_tmp.json", help="Temporary chunks file path")
    p.add_argument("--per-chunk-out", default="per_chunk_results.json", help="Per-chunk results file")
    p.add_argument("--final-out", default="final_report.json", help="Final aggregator output file")
    # Pass-through args for call_valueguard_api.py
    p.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"), help="LLM provider for per-chunk and aggregator")
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"), help="Model/engine name")
    p.add_argument("--concurrency", type=int, default=2, help="Concurrency for per-chunk LLM calls (passed to call_valueguard_api.py)")
    p.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds for aggregator call")
    args = p.parse_args()

    try:
        # Step 1: produce chunks (either from remote repo or local)
        if args.repo:
            # call generate_chunks.py --remote <repo> --out <chunks-out> --depth <depth>
            run_subprocess([GENERATE_SCRIPT, "--remote", args.repo, "--out", args.chunks_out, "--depth", str(args.depth)])
        else:
            # assume chunks.json exists in cwd; copy to chunks_out if different
            default_chunks = "chunks.json"
            if not os.path.exists(default_chunks):
                raise RuntimeError("No --repo specified and default chunks.json not found.")
            if args.chunks_out != default_chunks:
                # copy
                with open(default_chunks, "r", encoding="utf-8") as fr:
                    data = fr.read()
                with open(args.chunks_out, "w", encoding="utf-8") as fw:
                    fw.write(data)
                print("Copied %s -> %s" % (default_chunks, args.chunks_out))

        # Step 2: optionally limit number of chunks
        all_chunks = load_json(args.chunks_out)
        if not isinstance(all_chunks, list):
            raise RuntimeError("chunks file must contain a JSON array")
        if len(all_chunks) == 0:
            raise RuntimeError("No chunks found to analyze.")
        limited = all_chunks[: args.num_chunks]
        # overwrite chunks_out with limited set to avoid sending excessive requests
        write_json(limited, args.chunks_out)
        print("Prepared %d chunks (limited to %d) in %s" % (len(limited), args.num_chunks, args.chunks_out))

        # Step 3: call call_valueguard_api.py to process per-chunk results
        # Build cmd: call_valueguard_api.py --chunks <chunks_out> --out <per_chunk_out> --provider <provider> --model <model> --concurrency <concurrency>
        call_cmd = [
            CALL_SCRIPT,
            "--chunks",
            args.chunks_out,
            "--out",
            args.per_chunk_out,
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--concurrency",
            str(args.concurrency),
        ]
        # preserve environment for API keys
        run_subprocess(call_cmd)

        # Step 4: load per-chunk results
        per_chunk_results = load_json(args.per_chunk_out)
        if not isinstance(per_chunk_results, list):
            raise RuntimeError("per-chunk results must be a JSON array")

        # Build aggregator input
        repo_name = args.repo or os.getenv("GITHUB_REPOSITORY", "local")
        pr_id = os.getenv("PR_NUMBER", "unknown")
        agg_input = {"repo": repo_name, "pr_id": pr_id, "chunks": per_chunk_results, "repo_metadata": {"README_summary": "", "changed_files_count": len(per_chunk_results)}}

        # Step 5: load aggregator prompt and call aggregator LLM
        if not os.path.exists(AGG_PROMPT):
            raise RuntimeError("Aggregator prompt not found at %s" % AGG_PROMPT)
        with open(AGG_PROMPT, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        print("Calling aggregator LLM (provider=%s model=%s) ..." % (args.provider, args.model))
        agg_output = call_aggregator_llm(args.provider, args.model, prompt_template, agg_input, timeout=args.timeout)

        # Step 6: validate aggregator output against schema
        if not os.path.exists(AGG_SCHEMA):
            raise RuntimeError("Aggregator schema not found at %s" % AGG_SCHEMA)
        try:
            validate_aggregator_output(agg_output, AGG_SCHEMA)
        except Exception as e:
            # Save raw aggregator output for inspection and raise
            write_json(agg_output, args.final_out + ".raw.json")
            raise

        # Step 7: write final output
        write_json(agg_output, args.final_out)
        print("Wrote final aggregator report to %s" % args.final_out)
    except Exception as exc:
        print("ERROR:", str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
