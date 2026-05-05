import re


VALID_CIPHER_NAMES = {
    "caesar", "atbash", "vigenere", "rot13",
    "rail_fence", "reverse", "base64", "morse", "bacon",
}

ALIAS_MAP = {
    "rot-13": "rot13",
    "rot 13": "rot13",
    "caesar cipher": "caesar",
    "caesar's cipher": "caesar",
    "base 64": "base64",
    "base-64": "base64",
    "rail fence cipher": "rail_fence",
    "rail fence": "rail_fence",
    "rail-fence": "rail_fence",
    "railfence": "rail_fence",
    "columnar transposition": "rail_fence",
    "columnar": "rail_fence",
    "transposition cipher": "rail_fence",
    "vigenere cipher": "vigenere",
    "vigenere's cipher": "vigenere",
    "vigenère": "vigenere",
    "vigenere": "vigenere",
    "bacon cipher": "bacon",
    "bacon's cipher": "bacon",
    "morse code": "morse",
    "morse-code": "morse",
    "atbash cipher": "atbash",
    "reversed": "reverse",
    "reversal": "reverse",
    "string reversal": "reverse",
    "reverse cipher": "reverse",
}


def prompt_baseline(ciphertext: str) -> str:
    return (
        f"What encryption method was used to produce the following ciphertext?\n\n"
        f"{ciphertext}\n\n"
        f"Answer with just the cipher name."
    )


def prompt_choices(ciphertext: str) -> str:
    return (
        f"What encryption method was used to produce the following ciphertext?\n\n"
        f"{ciphertext}\n\n"
        f"Choose one: Caesar, Atbash, Vigenere, ROT13, Rail Fence, Reverse, Base64, Morse Code, Bacon Cipher\n\n"
        f"Answer with just the cipher name."
    )


def prompt_cot(ciphertext: str) -> str:
    return (
        f"What encryption method was used to produce the following ciphertext? "
        f"Think step by step about the patterns you observe, then give your final answer.\n\n"
        f"{ciphertext}\n\n"
        f"Analysis:"
    )


def prompt_expert(ciphertext: str) -> str:
    return (
        f"You are an expert cryptanalyst. Examine the following ciphertext and identify the "
        f"encryption method used. Consider letter frequencies, character set, structure, and "
        f"any visible patterns.\n\n"
        f"{ciphertext}\n\n"
        f"Cipher type:"
    )


PROMPT_VARIANTS = {
    "baseline": prompt_baseline,
    "choices":  prompt_choices,
    "cot":      prompt_cot,
}



def _extract_cipher(text: str) -> str | None:
    """Try to extract a cipher name from a short text snippet."""
    text = text.strip().lower()
    text = re.split(r'[\n.,(]', text)[0].strip()
    if text in VALID_CIPHER_NAMES:
        return text
    for alias, canonical in ALIAS_MAP.items():
        if alias in text:
            return canonical
    for name in VALID_CIPHER_NAMES:
        if name in text:
            return name
    return None


def parse_answer(raw_response: str) -> tuple[str | None, bool]:
    """
    Extract the cipher name from the model's response.

    Returns:
        (normalized_cipher_name, parseable)
        If not parseable, returns (raw_response_snippet, False).
    """
    full = raw_response.strip().lower()

    # 1. Look for explicit answer markers (search from the end for last occurrence)
    for marker in ("final answer:", "the answer is", "answer:", "cipher type:", "cipher:",
                   "the cipher is", "this is a", "this appears to be", "this is"):
        idx = full.rfind(marker)
        if idx != -1:
            result = _extract_cipher(full[idx + len(marker):])
            if result:
                return result, True

    # 2. Check the last 3 lines (answer often at end of long responses)
    lines = [l.strip() for l in full.splitlines() if l.strip()]
    for line in reversed(lines[-3:]):
        result = _extract_cipher(line)
        if result:
            return result, True

    # 3. Check the first line
    if lines:
        result = _extract_cipher(lines[0])
        if result:
            return result, True

    # 4. Scan entire response for any cipher name mention
    for alias, canonical in ALIAS_MAP.items():
        if alias in full:
            return canonical, True
    for name in VALID_CIPHER_NAMES:
        if name in full:
            return name, True

    # Unparseable
    snippet = raw_response[:80].replace('\n', ' ')
    return snippet, False
