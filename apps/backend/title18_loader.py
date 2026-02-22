"""
Load Title 18 US Criminal Code from usc18 2.xml for Groq context.
Used for criminal analysis only (not semantic search). Groq picks applicable section.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

# Default path to Title 18 XML in backend
DEFAULT_XML_PATH = Path(__file__).resolve().parent / "usc18 2.xml"
USLM_NS = "http://xml.house.gov/schemas/uslm/1.0"
NS = {"u": USLM_NS}
# Max chars to pass to Groq (leave room for transcript + prosocial chunks)
MAX_TITLE18_CHARS = 24_000
# Max number of sections to include (early sections are often most relevant)
MAX_SECTIONS = 100


def _text_of(el) -> str:
    if el is None:
        return ""
    return " ".join((el.text or "") + " ".join(_text_of(c) for c in el) + (el.tail or "")).strip()


def load_title18_sections(
    xml_path: Optional[Path] = None,
    max_chars: int = MAX_TITLE18_CHARS,
    max_sections: int = MAX_SECTIONS,
) -> str:
    """
    Parse Title 18 XML and return a single string of section summaries for Groq.
    Each section is formatted as "18 USC § <num>: <heading> <content snippet>".
    """
    path = xml_path or DEFAULT_XML_PATH
    if not path.exists():
        return "Title 18 (U.S. Criminal Code) reference not available."

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return "Title 18 (U.S. Criminal Code) could not be parsed."

    sections: List[Dict[str, str]] = []
    total_chars = 0

    for section in root.iter(f"{{{USLM_NS}}}section"):
        if section.get("status") == "repealed":
            continue
        num_el = section.find(f"{{{USLM_NS}}}num")
        num = num_el.get("value", "") if num_el is not None else ""
        heading_el = section.find(f"{{{USLM_NS}}}heading")
        heading = _text_of(heading_el) if heading_el is not None else ""
        content_el = section.find(f"{{{USLM_NS}}}content")
        content = _text_of(content_el) if content_el is not None else ""
        # First 200 chars of content to keep context size down
        content_snip = (content[:200] + "…") if len(content) > 200 else content
        line = f"18 USC § {num}: {heading}. {content_snip}"
        sections.append({"num": num, "text": line})
        total_chars += len(line) + 1
        if len(sections) >= max_sections or total_chars >= max_chars:
            break

    if not sections:
        return "Title 18 (U.S. Criminal Code) sections could not be extracted."
    return "\n".join(s["text"] for s in sections)


# Cached string for runtime (avoid re-parsing on every request)
_title18_context: Optional[str] = None


def get_title18_context(force_reload: bool = False) -> str:
    """Return Title 18 context string for Groq. Cached after first load."""
    global _title18_context
    if _title18_context is None or force_reload:
        _title18_context = load_title18_sections()
    return _title18_context
