# !/usr/bin/env python3
"""Call LLM per-chunk, validate against per-chunk schema, and write per_chunk_results.json


Requires:
- python3
- pip install requests jsonschema


Environment variables:
- LLM_API_KEY (or set in code)
- LLM_API_URL (optional)


This is intentionally generic so you can adapt to your LLM provider.
"""
import json
import os
import sys
import time
import requests
from jsonschema import validate, ValidationError


PER_CHUNK_PROMPT_PATH = os.path.join("prompts", "per_chunk_prompt.txt")
PER_CHUNK_SCHEMA = os.path.join("schemas", "per_chunk_schema.json")

# Load prompt template
with open(PER_CHUNK_PROMPT_PATH, "r", encoding="utf-8") as f:
    PER_CHUNK_PROMPT = f.read()


with open(PER_CHUNK_SCHEMA, "r", encoding="utf-8") as f:
    PER_CHUNK_SCHEMA_JSON = json.load(f)




def call_llm(payload: dict) -> str:
    """Generic HTTP POST to LLM API.
    Modify this function to match your provider's expected shape.
    It should return the assistant text as a string.
    """
    api_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "messages": payload,
    "temperature": 0.0,
    "max_tokens": int(os.getenv("PER_CHUNK_MAX_TOKENS", "900"))
    }
    resp = requests.post(api_url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # provider-specific: extract text
    # for OpenAI-like: data['choices'][0]['message']['content']
    return data['choices'][0]['message']['content']

def build_messages(chunk: dict) -> list:
    # Inject chunk JSON into the user prompt placeholder
    user_content = PER_CHUNK_PROMPT.replace("{{INPUT_JSON}}", json.dumps(chunk, ensure_ascii=False))
    return [
    {"role": "system", "content": "You are ValueGuard Assistant. Return only JSON matching schema."},
    {"role": "user", "content": user_content}
    ]




def validate_per_chunk_output(obj: dict):
    try:
        validate(instance=obj, schema=PER_CHUNK_SCHEMA_JSON)
        return True, None
    except ValidationError as e:
        return False, str(e)




def main():
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CHUNKS_IN", "chunks.json")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PER_CHUNK_OUT", "per_chunk_results.json")


    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Calling LLM on {len(chunks)} chunks...")

    results = []
    for chunk in chunks:
        messages = build_messages(chunk)
        try:
            text = call_llm(messages)
        except Exception as e:
            # fallback: mark for manual review
            results.append({
            "chunk_id": chunk.get("chunk_id"),
            "findings": [],
            "summary": f"error calling llm: {e}"
            })
            continue


        # try to parse JSON
        parsed = None
        for attempt in range(2):
            try:
                parsed = json.loads(text)
                break
            except json.JSONDecodeError:
                # attempt to repair: strip leading/trailing garbage lines
                text = text.strip()
                # naive: find first { and last }
                i = text.find("{")
                j = text.rfind("}")
                if i != -1 and j != -1 and j > i:
                    candidate = text[i:j+1]
                    try:
                        parsed = json.loads(candidate)
                        break
                    except Exception:
                        pass
                # else give up
                parsed = None
                break


        if parsed is None:
            results.append({
            "chunk_id": chunk.get("chunk_id"),
            "findings": [],
            "summary": "invalid_json_from_model"
            })
            continue


        ok, err = validate_per_chunk_output(parsed)
        if not ok:
            # best-effort: try to salvage or mark for review
            results.append({
            "chunk_id": chunk.get("chunk_id"),
            "findings": parsed.get("findings", []),
            "summary": f"schema_validation_failed: {err}"
            })
        else:
            results.append(parsed)
        # rate limit friendly
        time.sleep(0.3)


    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote per-chunk results to {out_path}")




if __name__ == '__main__':
    main()