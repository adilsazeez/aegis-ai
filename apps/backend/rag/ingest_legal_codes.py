import re
from pathlib import Path

from lxml import etree

from config import get_settings, get_actian_client
from rag.embeddings import embed_batch

NS = {"uslm": "http://xml.house.gov/schemas/uslm/1.0"}

THREAT_RELEVANT_CHAPTERS = {
    "5": "arson",
    "7": "assault",
    "11A": "child_support",
    "18": "congressional_threats",
    "19": "conspiracy",
    "25": "counterfeiting",
    "41": "extortion_and_threats",
    "42": "extortionate_credit",
    "43": "false_personation",
    "44": "firearms",
    "51": "homicide",
    "55": "kidnapping",
    "63": "mail_fraud",
    "71": "obscenity",
    "73": "obstruction_of_justice",
    "74": "partial_birth_abortions",
    "77": "peonage_slavery",
    "84": "presidential_threats",
    "88": "privacy",
    "89": "professions_and_occupations",
    "95": "racketeering",
    "96": "racketeer_influenced",
    "103": "robbery_and_burglary",
    "109A": "sexual_abuse",
    "110": "sexual_exploitation_children",
    "110A": "domestic_violence_stalking",
    "113": "stolen_property",
    "113B": "terrorism",
    "116": "trafficking",
    "117": "transportation_for_illegal_sex",
    "119": "wire_electronic_surveillance",
}


def _extract_text(element) -> str:
    """Recursively extract all text from an XML element, stripping tags."""
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(parts)


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove artifacts."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def parse_title18_xml(xml_path: str) -> list[dict]:
    """
    Parse the Title 18 XML file and extract threat-relevant sections.
    Returns a list of chunk dicts with metadata.
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()
    chunks = []

    chapters = root.findall(".//uslm:chapter", NS)

    for chapter in chapters:
        chapter_num_el = chapter.find("uslm:num", NS)
        chapter_heading_el = chapter.find("uslm:heading", NS)

        if chapter_num_el is None:
            continue

        chapter_num = chapter_num_el.get("value", "")
        if chapter_num not in THREAT_RELEVANT_CHAPTERS:
            continue

        category = THREAT_RELEVANT_CHAPTERS[chapter_num]
        chapter_heading = _clean_text(_extract_text(chapter_heading_el)) if chapter_heading_el is not None else ""
        chapter_label = f"Chapter {chapter_num} - {chapter_heading}"

        sections = chapter.findall(".//uslm:section", NS)

        for section in sections:
            if section.get("status") == "repealed":
                continue

            section_num_el = section.find("uslm:num", NS)
            section_heading_el = section.find("uslm:heading", NS)

            if section_num_el is None:
                continue

            section_num = section_num_el.get("value", "")
            section_heading = _clean_text(_extract_text(section_heading_el)) if section_heading_el is not None else ""

            content_parts = []
            for content_el in section.findall(".//uslm:content", NS):
                text = _clean_text(_extract_text(content_el))
                if text:
                    content_parts.append(text)

            for subsection in section.findall(".//uslm:subsection", NS):
                sub_content = _clean_text(_extract_text(subsection))
                if sub_content and sub_content not in " ".join(content_parts):
                    content_parts.append(sub_content)

            content = " ".join(content_parts)
            if not content or len(content) < 50:
                continue

            # Truncate very long sections to keep chunks within embedding model's sweet spot
            if len(content) > 2000:
                content = content[:2000]

            title = f"18 U.S.C. Section {section_num} - {section_heading}"

            chunks.append({
                "title": title,
                "section_number": section_num,
                "chapter": chapter_label,
                "category": category,
                "content": content,
                "search_text": f"{title}. {chapter_label}. {content}",
            })

    return chunks


async def ingest_into_actian(chunks: list[dict]) -> int:
    """
    Embed all chunks and store them in Actian VectorAI DB.
    Creates the collection if it doesn't exist.
    Returns the number of chunks ingested.
    """
    settings = get_settings()
    client = await get_actian_client()

    collection_name = settings.LEGAL_CODES_COLLECTION
    dimension = settings.EMBEDDING_DIMENSION

    await client.recreate_collection(name=collection_name, dimension=dimension)

    search_texts = [chunk["search_text"] for chunk in chunks]
    embeddings = embed_batch(search_texts)

    batch_size = 50
    total_ingested = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        ids = list(range(i, i + len(batch_chunks)))
        payloads = [
            {
                "title": c["title"],
                "section_number": c["section_number"],
                "chapter": c["chapter"],
                "category": c["category"],
                "content": c["content"],
            }
            for c in batch_chunks
        ]

        await client.batch_upsert(
            collection_name,
            ids=ids,
            vectors=batch_embeddings,
            payloads=payloads,
        )
        total_ingested += len(batch_chunks)

    await client.flush(collection_name)
    return total_ingested


async def run_ingestion() -> dict:
    """Full ingestion pipeline: parse XML -> embed -> store in Actian."""
    settings = get_settings()
    xml_path = str(Path(__file__).parent.parent / "data" / "usc18.xml")

    chunks = parse_title18_xml(xml_path)
    if not chunks:
        return {"chunks_ingested": 0, "collection_name": settings.LEGAL_CODES_COLLECTION, "message": "No chunks found"}

    count = await ingest_into_actian(chunks)

    return {
        "chunks_ingested": count,
        "collection_name": settings.LEGAL_CODES_COLLECTION,
        "message": f"Successfully ingested {count} legal code sections into Actian VectorAI DB",
    }
