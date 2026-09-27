"""
Phase 2: Text Normalization
===========================
Country-aware normalization for business names and addresses.
Handles US, India, and France (generalizes for unseen countries).
"""

import re
import unicodedata
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEGAL_SUFFIX_MAP = {
    # Corporate
    "corporation": "inc", "incorporated": "inc", "corp": "inc",
    "inc": "inc", "co": "co",
    # Private
    "private": "pvt", "pvt": "pvt",
    # Limited
    "limited": "ltd", "ltd": "ltd",
    # Partnership / LLC
    "llc": "llc", "llp": "llc", "lp": "llc",
    "partnership": "llc",
    # India-specific
    "enterprises": "enterprises", "enterprise": "enterprises",
    "traders": "traders", "trader": "traders",
    "brothers": "brothers", "bros": "brothers",
    "associates": "associates", "assoc": "associates",
    "industries": "industries", "industry": "industries",
    "solutions": "solutions",
    # General
    "group": "group", "holdings": "holdings",
    "international": "intl", "intl": "intl",
    "services": "services", "service": "services",
    "technologies": "tech", "technology": "tech", "tech": "tech",
}

STOPWORDS = {"the", "a", "an", "of", "and", "in", "at", "for", "to", "by", "with"}

US_STREET_ABBR = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bdr\b": "drive", r"\bln\b": "lane",
    r"\bct\b": "court", r"\bpl\b": "place", r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway", r"\bsq\b": "square", r"\bfte\b": "suite",
    r"\bste\b": "suite", r"\bapt\b": "apartment", r"\bfl\b": "floor",
    r"\bn\b": "north", r"\bs\b": "south", r"\be\b": "east", r"\bw\b": "west",
}

INDIA_STREET_ABBR = {
    r"\bnr\b": "near", r"\bopp\b": "opposite", r"\bsoc\b": "society",
}

GENERAL_ABBR = {
    r"\&": "and",
}

# Written numbers 1-10 → digits
WRITTEN_NUMBERS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# Regex for PIN/ZIP codes
US_ZIP_RE = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
IN_PIN_RE = re.compile(r'\b(\d{6})\b')
FR_POST_RE = re.compile(r'\b(\d{5})\b')

# DBA markers
DBA_RE = re.compile(
    r'\b(dba|d/b/a|doing business as|trading as|t/a|formerly|fka|also known as)\b',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Core Utilities
# ---------------------------------------------------------------------------

def unicode_normalize(text: str) -> str:
    """Normalize unicode, convert to ASCII best-effort."""
    text = unicodedata.normalize('NFKD', text)
    return text.encode('ascii', 'ignore').decode('ascii')


def remove_punctuation(text: str) -> str:
    """Remove common punctuation except spaces."""
    return re.sub(r"[.,\-'\"()\[\]{}|\\/:;!?@#$%^*_+=<>`~]", " ", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def extract_dba_parts(name: str) -> Tuple[str, Optional[str]]:
    """
    Split 'Legal Name DBA Trade Name' into (legal_name, trade_name).
    Returns (original, None) if no DBA marker found.
    """
    m = DBA_RE.search(name)
    if m:
        legal = name[:m.start()].strip()
        trade = name[m.end():].strip()
        return legal or name, trade or None
    return name, None


# ---------------------------------------------------------------------------
# Business Name Normalization
# ---------------------------------------------------------------------------

def normalize_business_name(name: str, country: str = "") -> dict:
    """
    Returns a dict with multiple normalized views of the business name:
      - raw_lower: lowercase only
      - normalized: full normalization
      - sorted_tokens: tokens sorted alphabetically (handles transpositions)
      - core: stopwords + legal suffix removed
      - trade_name: trade name after DBA split (or None)
    """
    if not isinstance(name, str):
        name = str(name) if name else ""

    # 0. Unicode → ASCII
    name = unicode_normalize(name)

    # 1. Lowercase
    name_lc = name.lower().strip()

    # 1b. Extract DBA parts before further normalization
    legal_part, trade_part = extract_dba_parts(name_lc)

    def _normalize_one(s: str) -> str:
        # 2. Replace & → and, handle co.
        s = re.sub(r'&', ' and ', s)
        s = re.sub(r'\bco\.\b', 'co', s)

        # 3. Expand general abbreviations
        for pat, repl in GENERAL_ABBR.items():
            s = re.sub(pat, repl, s, flags=re.IGNORECASE)

        # 4. Remove punctuation
        s = remove_punctuation(s)

        # 5. Collapse whitespace
        s = collapse_whitespace(s)

        # 6. Standardize legal suffixes
        tokens = s.split()
        normalized_tokens = []
        for tok in tokens:
            normalized_tokens.append(LEGAL_SUFFIX_MAP.get(tok, tok))

        return " ".join(normalized_tokens)

    normalized = _normalize_one(legal_part)
    trade_normalized = _normalize_one(trade_part) if trade_part else None

    # 7. Sorted tokens (handles word-order transpositions)
    tokens = normalized.split()
    sorted_tokens = " ".join(sorted(tokens))

    # 8. Core: remove stopwords and legal suffixes
    core_tokens = [
        t for t in tokens
        if t not in STOPWORDS and t not in set(LEGAL_SUFFIX_MAP.values())
    ]
    core = " ".join(core_tokens)

    return {
        "raw_lower": name_lc,
        "normalized": normalized,
        "sorted_tokens": sorted_tokens,
        "core": core,
        "trade_name": trade_normalized,
        "tokens": tokens,
        "core_tokens": core_tokens,
    }


# ---------------------------------------------------------------------------
# Address Normalization
# ---------------------------------------------------------------------------

def extract_zip_pin(address: str, country: str = "") -> Optional[str]:
    """Extract ZIP (US), PIN (India), or postal code (France)."""
    if not address:
        return None
    country_upper = (country or "").upper()
    if "US" in country_upper or "UNITED STATES" in country_upper:
        m = US_ZIP_RE.search(address)
        return m.group(1) if m else None
    elif "IN" in country_upper or "INDIA" in country_upper:
        m = IN_PIN_RE.search(address)
        return m.group(1) if m else None
    elif "FR" in country_upper or "FRANCE" in country_upper:
        m = FR_POST_RE.search(address)
        return m.group(1) if m else None
    else:
        # Try all patterns
        for pat in [US_ZIP_RE, IN_PIN_RE, FR_POST_RE]:
            m = pat.search(address)
            if m:
                return m.group(1)
        return None


def _apply_abbreviations(text: str, abbr_map: dict) -> str:
    for pat, repl in abbr_map.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def normalize_address(address: str, country: str = "") -> dict:
    """
    Normalize a business address.
    Returns dict with:
      - normalized: fully normalized address string
      - zip_pin: extracted ZIP/PIN code (or None)
      - tokens: list of address tokens
    """
    if not isinstance(address, str):
        address = str(address) if address else ""

    # 0. Unicode → ASCII
    address = unicode_normalize(address)

    # 1. Lowercase
    addr = address.lower().strip()

    # 2. Extract ZIP/PIN before normalization alters digits
    zip_pin = extract_zip_pin(addr, country)

    # 3. Replace & → and
    addr = re.sub(r'&', ' and ', addr)

    # 4. Apply general abbreviations
    for pat, repl in GENERAL_ABBR.items():
        addr = re.sub(pat, repl, addr, flags=re.IGNORECASE)

    # 5. Country-specific abbreviation expansion
    country_upper = (country or "").upper()
    if "US" in country_upper or "UNITED STATES" in country_upper:
        addr = _apply_abbreviations(addr, US_STREET_ABBR)
    elif "IN" in country_upper or "INDIA" in country_upper:
        addr = _apply_abbreviations(addr, INDIA_STREET_ABBR)
    else:
        # Apply all (safe for France and unknown countries)
        addr = _apply_abbreviations(addr, US_STREET_ABBR)
        addr = _apply_abbreviations(addr, INDIA_STREET_ABBR)

    # 6. Convert written numbers to digits
    for word, digit in WRITTEN_NUMBERS.items():
        addr = re.sub(r'\b' + word + r'\b', digit, addr)

    # 7. Remove punctuation
    addr = remove_punctuation(addr)

    # 8. Collapse whitespace
    addr = collapse_whitespace(addr)

    tokens = addr.split()

    return {
        "normalized": addr,
        "zip_pin": zip_pin,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Country-aware dispatcher
# ---------------------------------------------------------------------------

def normalize_record(business_name: str, business_address: str, country: str) -> dict:
    """
    Full normalization for a single record.
    Dispatches to country-specific logic internally.
    Returns a flat dict of all normalized fields.
    """
    name_info = normalize_business_name(business_name, country)
    addr_info = normalize_address(business_address, country)

    return {
        # Name fields
        "name_raw_lower": name_info["raw_lower"],
        "name_normalized": name_info["normalized"],
        "name_sorted": name_info["sorted_tokens"],
        "name_core": name_info["core"],
        "name_trade": name_info["trade_name"],
        "name_tokens": name_info["tokens"],
        "name_core_tokens": name_info["core_tokens"],
        # Address fields
        "addr_normalized": addr_info["normalized"],
        "addr_tokens": addr_info["tokens"],
        "zip_pin": addr_info["zip_pin"],
        # Country
        "country_norm": country.strip().upper() if country else "",
    }
