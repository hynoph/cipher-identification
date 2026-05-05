# Cipher Type Identification by LLMs — Zero-Shot Evaluation

Can LLMs identify cipher types from ciphertext alone, without any training?

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys, then:
```bash
export $(cat .env | xargs)
```

## Workflow

```bash
# 1. Generate the dataset (1,372 samples: 98 plaintexts × 14 cipher configs)
python data/generate_dataset.py

# 2. Sanity check — encrypt "HELLO WORLD" with all ciphers
python -c "
import data.generate_dataset as g
text = 'HELLO WORLD'
for cfg in g.CIPHER_CONFIGS:
    print(f\"{cfg['cipher_type']:12} {cfg['cipher_params']} => {cfg['fn'](text)}\")
"

# 3. Dry run — inspect prompts without calling any API
python -m eval.run_eval --dry-run --limit 3

# 4. Small test run — 1 model, 1 prompt, 10 samples
python -m eval.run_eval --model gpt-4o --prompt baseline --limit 10

# 5. Full evaluation
python -m eval.run_eval

# 6. Score results
python -m eval.score

# 7. Generate figures and tables
python -m eval.analyze
```

Results land in `results/`. Figures are in `results/figures/`.

## Cipher types covered (14 configurations, 9 canonical labels)

| Label       | Category      | Variants tested              |
|-------------|---------------|------------------------------|
| caesar      | substitution  | shift 3, 7, 19               |
| atbash      | substitution  | —                            |
| vigenere    | substitution  | keys: ACL, CIPHER, SECRET    |
| rot13       | substitution  | (Caesar shift=13)            |
| rail_fence  | transposition | 2 rails, 3 rails             |
| reverse     | transposition | —                            |
| base64      | encoding      | —                            |
| morse       | encoding      | —                            |
| bacon       | encoding      | —                            |

## Key hypotheses

1. LLMs achieve above-chance accuracy on cipher identification zero-shot
2. ROT13 accuracy is significantly higher than other Caesar shifts (memorization effect)
3. Encoding ciphers (Base64, Morse) are easier to identify than substitution ciphers
4. Providing a fixed list of choices improves accuracy over open-ended identification
5. Reasoning models outperform chat models
6. Longer ciphertexts are easier to identify than shorter ones
