"""
Identity term detection for Hindi/Hinglish text.
Lexicon covers four categories: Religion, Caste, Gender, Region (Table 3 in paper).
"""

import re
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Identity Lexicon (Table 3)
# Each category maps to a list of terms (Hindi + Hinglish / transliterated).
# All terms are lowercased for matching.
# ---------------------------------------------------------------------------

IDENTITY_LEXICON: Dict[str, List[str]] = {
    "religion": [
        # Hindi (Devanagari)
        "हिंदू", "हिन्दू", "मुस्लिम", "मुसलमान", "ईसाई", "क्रिश्चियन",
        "सिख", "जैन", "बौद्ध", "बौध", "इस्लाम", "इस्लामी",
        # Hinglish / Romanised
        "hindu", "muslim", "musalman", "christian", "isai", "sikh",
        "jain", "buddhist", "baudh", "islam", "islamic", "mulla",
        "maulana", "pandit", "padre",
    ],
    "caste": [
        # Hindi (Devanagari)
        "दलित", "ब्राह्मण", "ब्राम्हण", "क्षत्रिय", "वैश्य", "शूद्र",
        "सवर्ण", "अछूत", "ओबीसी", "अनुसूचित",
        # Hinglish / Romanised
        "dalit", "brahmin", "brahman", "kshatriya", "vaishya", "shudra",
        "savarna", "achhoot", "obc", "chamar", "bhangi", "schedule caste",
        "scheduled caste", "upper caste", "lower caste",
    ],
    "gender": [
        # Hindi (Devanagari)
        "महिला", "पुरुष", "औरत", "आदमी", "लड़की", "लड़का",
        "स्त्री", "नारी", "मर्द", "पुरुषों", "महिलाओं", "औरतों",
        # Hinglish / Romanised
        "woman", "women", "man", "men", "mahila", "purush", "aurat",
        "ladki", "ladka", "mard", "stri", "naari", "female", "male",
    ],
    "region": [
        # Hindi (Devanagari)
        "बिहारी", "पंजाबी", "मद्रासी", "बंगाली", "मराठी", "गुजराती",
        "तमिल", "केरलाइट", "उत्तर भारतीय", "दक्षिण भारतीय",
        # Hinglish / Romanised
        "bihari", "punjabi", "madrasi", "bengali", "marathi", "gujarati",
        "tamil", "keralite", "north indian", "south indian", "bhaiya",
        "chinki", "nepali", "kashmiri",
    ],
}

# Flatten for quick lookup
ALL_TERMS: Set[str] = set()
TERM_TO_CATEGORY: Dict[str, str] = {}
for _cat, _terms in IDENTITY_LEXICON.items():
    for _t in _terms:
        ALL_TERMS.add(_t.lower())
        TERM_TO_CATEGORY[_t.lower()] = _cat


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def detect_identity_terms(text: str) -> Dict[str, List[str]]:
    """
    Return dict {category: [matched_terms]} for a single text.
    Uses simple substring matching after normalisation.
    """
    normed = _normalise(text)
    found: Dict[str, List[str]] = {}
    for term in ALL_TERMS:
        if term in normed:
            cat = TERM_TO_CATEGORY[term]
            found.setdefault(cat, []).append(term)
    return found


def has_identity_mention(text: str) -> bool:
    """Return True if at least one identity term is present."""
    return len(detect_identity_terms(text)) > 0


def get_identity_group(text: str) -> str:
    """
    Return the *primary* identity category for a text.
    If multiple categories match, return the first by priority:
    religion > caste > gender > region.
    If none match, return 'none'.
    """
    found = detect_identity_terms(text)
    for cat in ["religion", "caste", "gender", "region"]:
        if cat in found:
            return cat
    return "none"


def get_swap_pairs() -> Dict[str, List[List[str]]]:
    """
    Return swap pairs for counterfactual generation.
    Each category has a list of term-groups that can be swapped with each other.
    """
    return {
        "religion": [
            ["hindu", "हिंदू", "हिन्दू"],
            ["muslim", "मुस्लिम", "मुसलमान"],
            ["christian", "ईसाई", "isai", "क्रिश्चियन"],
            ["sikh", "सिख"],
            ["jain", "जैन"],
            ["buddhist", "बौद्ध", "baudh"],
        ],
        "caste": [
            ["brahmin", "ब्राह्मण", "brahman"],
            ["dalit", "दलित"],
            ["kshatriya", "क्षत्रिय"],
            ["vaishya", "वैश्य"],
            ["obc", "ओबीसी"],
        ],
        "gender": [
            ["woman", "women", "महिला", "औरत", "ladki", "लड़की", "female"],
            ["man", "men", "पुरुष", "आदमी", "ladka", "लड़का", "male"],
        ],
        "region": [
            ["bihari", "बिहारी", "bhaiya"],
            ["punjabi", "पंजाबी"],
            ["madrasi", "मद्रासी"],
            ["bengali", "बंगाली"],
            ["tamil", "तमिल"],
        ],
    }


if __name__ == "__main__":
    # Quick sanity check
    samples = [
        "Muslims are peaceful people",
        "ये हिंदू लोग बहुत अच्छे हैं",
        "Bihari log hardworking hote hain",
        "This is a normal tweet with no identity terms",
    ]
    for s in samples:
        print(f"Text: {s}")
        print(f"  Identity terms: {detect_identity_terms(s)}")
        print(f"  Group: {get_identity_group(s)}")
        print()
