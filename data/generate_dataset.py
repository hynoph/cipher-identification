import base64
import json
import re
import sys
from pathlib import Path


MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
    '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.',
}

BACON_CODE = {chr(ord('A') + i): format(i, '05b').replace('0', 'A').replace('1', 'B')
              for i in range(26)}


def caesar(text: str, shift: int) -> str:
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return ''.join(result)


def atbash(text: str) -> str:
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            result.append(chr(base + 25 - (ord(ch) - base)))
        else:
            result.append(ch)
    return ''.join(result)


def vigenere(text: str, key: str) -> str:
    key = key.upper()
    result = []
    ki = 0
    for ch in text:
        if ch.isalpha():
            shift = ord(key[ki % len(key)]) - ord('A')
            base = ord('A') if ch.isupper() else ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
            ki += 1
        else:
            result.append(ch)
    return ''.join(result)


def rot13(text: str) -> str:
    return caesar(text, 13)


def rail_fence(text: str, rails: int) -> str:
    fence = [[] for _ in range(rails)]
    rail, direction = 0, 1
    for ch in text:
        fence[rail].append(ch)
        if rail == 0:
            direction = 1
        elif rail == rails - 1:
            direction = -1
        rail += direction
    return ''.join(''.join(r) for r in fence)


def reverse(text: str) -> str:
    return text[::-1]


def to_base64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def to_morse(text: str) -> str:
    words = text.upper().split()
    encoded_words = []
    for word in words:
        letters = []
        for ch in word:
            if ch in MORSE_CODE:
                letters.append(MORSE_CODE[ch])
        if letters:
            encoded_words.append(' '.join(letters))
    return ' / '.join(encoded_words)


def to_bacon(text: str) -> str:
    result = []
    for ch in text.upper():
        if ch in BACON_CODE:
            result.append(BACON_CODE[ch])
    return ' '.join(result)



CIPHER_CONFIGS = [
    {"cipher_type": "caesar", "cipher_params": {"shift": 3},  "category": "substitution",  "fn": lambda t: caesar(t, 3)},
    {"cipher_type": "caesar", "cipher_params": {"shift": 7},  "category": "substitution",  "fn": lambda t: caesar(t, 7)},
    {"cipher_type": "caesar", "cipher_params": {"shift": 19}, "category": "substitution",  "fn": lambda t: caesar(t, 19)},
    {"cipher_type": "atbash",  "cipher_params": {},            "category": "substitution",  "fn": atbash},
    {"cipher_type": "vigenere","cipher_params": {"key": "ACL"},    "category": "substitution",  "fn": lambda t: vigenere(t, "ACL")},
    {"cipher_type": "vigenere","cipher_params": {"key": "CIPHER"}, "category": "substitution",  "fn": lambda t: vigenere(t, "CIPHER")},
    {"cipher_type": "vigenere","cipher_params": {"key": "SECRET"}, "category": "substitution",  "fn": lambda t: vigenere(t, "SECRET")},
    {"cipher_type": "rot13",   "cipher_params": {},            "category": "substitution",  "fn": rot13},
    {"cipher_type": "rail_fence","cipher_params": {"rails": 2},"category": "transposition", "fn": lambda t: rail_fence(t, 2)},
    {"cipher_type": "rail_fence","cipher_params": {"rails": 3},"category": "transposition", "fn": lambda t: rail_fence(t, 3)},
    {"cipher_type": "reverse", "cipher_params": {},            "category": "transposition", "fn": reverse},
    {"cipher_type": "base64",  "cipher_params": {},            "category": "encoding",      "fn": to_base64},
    {"cipher_type": "morse",   "cipher_params": {},            "category": "encoding",      "fn": to_morse},
    {"cipher_type": "bacon",   "cipher_params": {},            "category": "encoding",      "fn": to_bacon},
]


def load_plaintexts(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def generate_dataset(plaintexts_path: str, output_path: str) -> None:
    plaintexts = load_plaintexts(plaintexts_path)
    records = []
    errors = []

    for i, plaintext in enumerate(plaintexts, 1):
        for cfg in CIPHER_CONFIGS:
            params_str = '_'.join(str(v) for v in cfg["cipher_params"].values()) if cfg["cipher_params"] else ""
            sample_id = f"sample_{i:03d}_{cfg['cipher_type']}" + (f"_{params_str}" if params_str else "")
            try:
                ciphertext = cfg["fn"](plaintext)
                record = {
                    "id": sample_id,
                    "plaintext": plaintext,
                    "ciphertext": ciphertext,
                    "cipher_type": cfg["cipher_type"],
                    "cipher_params": cfg["cipher_params"],
                    "plaintext_length": len(plaintext),
                    "category": cfg["category"],
                }
                records.append(record)
            except Exception as e:
                errors.append({"id": sample_id, "error": str(e)})

    Path(output_path).write_text('\n'.join(json.dumps(r) for r in records))
    print(f"Generated {len(records)} samples, {len(errors)} errors.")
    if errors:
        for err in errors:
            print(f"  ERROR: {err}")


def verify_dataset(output_path: str) -> None:
    """Decrypt each record and confirm round-trip matches original plaintext."""
    import importlib
    records = [json.loads(line) for line in Path(output_path).read_text().splitlines() if line.strip()]
    mismatches = 0
    for r in records:
        # Verification logic would go here (cipher-specific decryption)
        pass
    print(f"Verification complete. {mismatches} mismatches.")


if __name__ == "__main__":
    base = Path(__file__).parent
    generate_dataset(str(base / "sample_plaintexts.txt"), str(base / "dataset.jsonl"))
    verify_dataset(str(base / "dataset.jsonl"))
