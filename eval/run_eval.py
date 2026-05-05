import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODELS
import config
from prompts.prompt_templates import PROMPT_VARIANTS, parse_answer



def call_openai(model_id: str, prompt: str) -> str:
    from openai import OpenAI
    import config
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    # Reasoning models may not support temperature
    is_reasoning = "o1" in model_id or "o3" in model_id
    kwargs = {} if is_reasoning else {"temperature": 0}
    token_param = "max_completion_tokens" if is_reasoning else "max_tokens"
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        **{token_param: 500},
        **kwargs,
    )
    return response.choices[0].message.content


def call_anthropic(model_id: str, prompt: str) -> str:
    from anthropic import Anthropic
    import config
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model_id,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response.content:
        return "[empty response]"
    return response.content[0].text


def call_together(model_id: str, prompt: str) -> str:
    from openai import OpenAI
    import config
    client = OpenAI(api_key=config.TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content


def call_deepseek(model_id: str, prompt: str) -> str:
    from openai import OpenAI
    import config
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    msg = response.choices[0].message
    # deepseek-reasoner puts the final answer in content, reasoning in reasoning_content
    # fall back to reasoning_content if content is empty
    content = msg.content or ""
    if not content.strip():
        content = getattr(msg, "reasoning_content", "") or ""
    return content


PROVIDER_CALLERS = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "together":  call_together,
    "deepseek":  call_deepseek,
}



def call_with_retry(provider: str, model_id: str, prompt: str, retries: int = 5) -> str:
    caller = PROVIDER_CALLERS[provider]
    delay = 30.0  # start with 30s on 429
    for attempt in range(retries + 1):
        try:
            return caller(model_id, prompt)
        except Exception as e:
            code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower()
            is_conn_err = "connection" in str(e).lower() or "getaddrinfo" in str(e).lower()
            if attempt < retries and (code in (429, 500, 503) or is_timeout or is_conn_err):
                print(f"  Rate limited. Waiting {delay}s before retry {attempt + 1}/{retries}...")
                time.sleep(delay)
                delay *= 2
            else:
                raise



def load_dataset(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def get_output_path(model_name: str, prompt_variant: str) -> Path:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    return out / f"{model_name}_{prompt_variant}.jsonl"


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def run_eval(model_cfg: dict, prompt_variant: str, dataset: list[dict], dry_run: bool, limit: int | None) -> None:
    model_name = model_cfg["name"]
    model_id   = model_cfg["model_id"]
    provider   = model_cfg["provider"]
    prompt_fn  = PROMPT_VARIANTS[prompt_variant]

    out_path = get_output_path(model_name, prompt_variant)
    done_ids = load_existing_ids(out_path)  # load once into memory

    samples = dataset[:limit] if limit else dataset
    samples = [s for s in samples if s["id"] not in done_ids]

    print(f"{model_name} / {prompt_variant} — {len(samples)} samples remaining")

    with open(out_path, "a") as fh:
        for i, sample in enumerate(samples):
            if sample["id"] in done_ids:  # in-memory check, no file re-read
                continue

            prompt = prompt_fn(sample["ciphertext"])

            if dry_run:
                if i < 3:
                    print(f"Sample {i+1} ({sample['id']}):")
                    print(prompt[:300])
                continue

            raw = call_with_retry(provider, model_id, prompt)
            pred, parseable = parse_answer(raw)
            correct = parseable and pred == sample["cipher_type"]

            record = {
                "id":               sample["id"],
                "cipher_type_true": sample["cipher_type"],
                "cipher_type_pred": pred,
                "raw_response":     raw,
                "parseable":        parseable,
                "correct":          correct,
                "model":            model_name,
                "prompt_variant":   prompt_variant,
            }
            fh.write(json.dumps(record) + "\n")
            done_ids.add(sample["id"])  # update in-memory set immediately
            fh.flush()
            print(f"  [{i + 1}/{len(samples)}] {sample['id']} -> {pred} ({'correct' if correct else 'wrong'})")

            time.sleep(config.DELAY_SECS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   help="Run only this model name")
    parser.add_argument("--prompt",  help="Run only this prompt variant")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling APIs")
    parser.add_argument("--limit",   type=int, help="Limit samples per run (for testing)")
    args = parser.parse_args()

    # Lock file — prevent two processes running simultaneously
    lock = ROOT / ".eval.lock"
    if lock.exists():
        print("Another eval process is already running (.eval.lock exists). Exiting.")
        sys.exit(1)
    lock.write_text(str(os.getpid()))
    try:
        _main(args)
    finally:
        lock.unlink(missing_ok=True)


def _main(args) -> None:
    dataset = load_dataset(str(ROOT / "data" / "dataset.jsonl"))

    models   = [m for m in MODELS if not args.model  or m["name"] == args.model]
    variants = [v for v in PROMPT_VARIANTS if not args.prompt or v == args.prompt]

    for model_cfg in models:
        for variant in variants:
            run_eval(model_cfg, variant, dataset, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
