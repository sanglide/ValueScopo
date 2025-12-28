#!/usr/bin/env python3
"""Aggregate per-chunk outputs and call aggregator prompt.


Outputs final_report.json
"""
import json
import os
import sys
import requests
from jsonschema import validate, ValidationError


AGG_PROMPT_PATH = os.path.join("prompts", "aggregator_prompt.txt")
AGG_SCHEMA_PATH = os.path.join("schemas", "aggregator_schema.json")


with open(AGG_PROMPT_PATH, "r", encoding="utf-8") as f:
    AGG_PROMPT = f.read()
with open(AGG_SCHEMA_PATH, "r", encoding="utf-8") as f:
    AGG_SCHEMA = json.load(f)




def call_llm(messages: list) -> str:
    api_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
    api_key = os.getenv("LLM_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "messages": messages,
    "temperature": 0.0,
    "max_tokens": int(os.getenv("AGG_MAX_TOKENS", "1600"))
    }
    resp = requests.post(api_url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content']




def build_messages(chunks: list, repo: str, pr_id: str):
    user_input = {"repo": repo, "pr_id": pr_id, "chunks": chunks, "repo_metadata": {"README_summary": "", "changed_files_count": len(chunks)}}
    content = AGG_PROMPT.replace("{{INPUT_JSON}}", json.dumps(user_input, ensure_ascii=False))
    return [{"role": "system", "content": "You are ValueGuard Aggregator. Return only JSON matching schema."}, {"role": "user", "content": content}]




def validate_final(obj: dict):
    try:
        validate(instance=obj, schema=AGG_SCHEMA)
        return True, None
    except ValidationError as e:
        return False, str(e)




def main():
    per_chunk_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PER_CHUNK_IN", "per_chunk_results.json")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FINAL_OUT", "final_report.json")
    with open(per_chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)


    repo = os.getenv("GITHUB_REPOSITORY", "repo")
    pr_id = os.getenv("PR_NUMBER", "unknown")


    messages = build_messages(chunks, repo, pr_id)
    text = call_llm(messages)


    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        # attempt salvage
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and j > i:
            parsed = json.loads(text[i:j+1])


    if parsed is None:
        print("Failed to parse aggregator response", file=sys.stderr)
        parsed = {"value_summary": {}, "suggested_fixes": [], "reviewer_requests": [], "overall_confidence": 0.0, "summary_text": "invalid_response"}


    ok, err = validate_final(parsed)
    if not ok:
        print("Aggregator schema validation failed:", err, file=sys.stderr)


    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    print(f"Wrote final report to {out_path}")




if __name__ == '__main__':
    main()