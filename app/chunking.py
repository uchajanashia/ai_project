import hashlib
import html
import re
import unicodedata
import uuid
from collections.abc import Iterable

import tiktoken

REQUIRED_COLUMNS = [
    "ბრძანება",
    "მიღების თარიღი",
    "დოკუმენტი",
    "დოკუმენტის ტიპი",
    "კატეგორია",
    "item_page_link",
    "Document_Date",
    "გადაწყვეტილება",
    "Company_Activity",
    "Complaint_Date",
    "Appeal_Information",
    "Penalty",
]

COLUMN_ALIASES: dict[str, list[str]] = {
    "ბრძანება": ["ბრძანება", "Document_Number", "document_number"],
    "მიღების თარიღი": ["მიღების თარიღი", "Document_Date", "Document_Date_1"],
    "დოკუმენტი": ["დოკუმენტი", "დოკუკმენტი", "item_page_title", "document_title"],
    "დოკუმენტის ტიპი": ["დოკუმენტის ტიპი", "Document_Type"],
    "კატეგორია": ["კატეგორია", "Tag", "category"],
    "item_page_link": ["item_page_link", "source_url", "web_scraper_start_url"],
    "Document_Date": ["Document_Date", "Document_Date_1", "მიღების თარიღი"],
    "გადაწყვეტილება": ["გადაწყვეტილება", "Decision", "Document_Description"],
    "Company_Activity": ["Company_Activity", "data3"],
    "Complaint_Date": ["Complaint_Date", "data_1"],
    "Appeal_Information": ["Appeal_Information", "Appeal_Procedure"],
    "Penalty": ["Penalty"],
}

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t\f\v]+")
MULTINEWLINE_RE = re.compile(r"\n{2,}")


def normalize_column_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.lstrip("\ufeff").strip()
    return normalized


def clean_text(raw_text: object) -> str:
    text = "" if raw_text is None else str(raw_text)
    if text.lower() == "nan":
        return ""
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_TAG_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def get_tokenizer(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def add_canonical_columns(df):
    frame = df.copy()
    normalized_map = {normalize_column_name(column): column for column in frame.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue

        source_name = None
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_map:
                source_name = normalized_map[normalized_alias]
                break
        if source_name is not None:
            frame[canonical] = frame[source_name]
    return frame


def validate_required_columns(df) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing)
            + ". Ensure data/infohub.csv contains all required fields."
        )


def sanitize_row(row: dict) -> dict:
    return {str(key): clean_text(value) for key, value in row.items()}


def _clip(value: str, max_chars: int | None) -> str:
    if max_chars is None:
        return value
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + " ..."


def build_structured_header(row: dict, for_chunk: bool = False) -> str:
    preview_limits = {
        "დოკუმენტი": 110,
        "გადაწყვეტილება": 130,
        "Company_Activity": 110,
        "Appeal_Information": 110,
        "Penalty": 80,
    } if for_chunk else {}

    document = _clip(row.get("დოკუმენტი", ""), preview_limits.get("დოკუმენტი"))
    decision = _clip(row.get("გადაწყვეტილება", ""), preview_limits.get("გადაწყვეტილება"))
    company_activity = _clip(
        row.get("Company_Activity", ""),
        preview_limits.get("Company_Activity"),
    )
    appeal_info = _clip(
        row.get("Appeal_Information", ""),
        preview_limits.get("Appeal_Information"),
    )
    penalty = _clip(row.get("Penalty", ""), preview_limits.get("Penalty"))

    return "\n".join(
        [
            f"დოკუმენტი: {document}",
            f"ბრძანება: {row.get('ბრძანება', '')}",
            f"დოკუმენტის ტიპი: {row.get('დოკუმენტის ტიპი', '')}",
            f"კატეგორია: {row.get('კატეგორია', '')}",
            f"მიღების თარიღი: {row.get('მიღების თარიღი', '')}",
            f"გადაწყვეტილება: {decision}",
            f"კომპანიის საქმიანობა: {company_activity}",
            f"საჩივრის თარიღი: {row.get('Complaint_Date', '')}",
            f"აპელაციის ინფორმაცია: {appeal_info}",
            f"სანქცია: {penalty}",
        ]
    ).strip()


def build_structured_full_text(row: dict) -> str:
    header = build_structured_header(row=row, for_chunk=False)
    body = row.get("გადაწყვეტილება", "")
    return f"{header}\n\n{body}".strip()


def _split_body_with_header(
    header_text: str,
    body_text: str,
    tokenizer,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> tuple[list[str], str]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be greater than zero.")

    header = header_text.strip()
    if not header:
        return [], ""

    separator = "\n\n"
    header_tokens = tokenizer.encode(header)
    separator_tokens = tokenizer.encode(separator)
    body_tokens = tokenizer.encode(body_text.strip()) if body_text.strip() else []

    max_header_tokens = max(160, chunk_size_tokens - (overlap_tokens + 220))
    if len(header_tokens) > max_header_tokens:
        header = tokenizer.decode(header_tokens[:max_header_tokens]).strip()
        header_tokens = tokenizer.encode(header)

    available_body_tokens = chunk_size_tokens - len(header_tokens) - len(separator_tokens)
    if available_body_tokens < max(120, overlap_tokens // 2):
        header_budget = max(120, chunk_size_tokens // 3)
        header = tokenizer.decode(header_tokens[:header_budget]).strip()
        header_tokens = tokenizer.encode(header)
        available_body_tokens = max(1, chunk_size_tokens - len(header_tokens) - len(separator_tokens))

    if not body_tokens:
        return [header], header

    effective_overlap = min(overlap_tokens, max(0, available_body_tokens - 1))
    step = max(1, available_body_tokens - effective_overlap)

    chunks: list[str] = []
    for start in range(0, len(body_tokens), step):
        body_slice = body_tokens[start : start + available_body_tokens]
        if not body_slice:
            continue
        chunk_body = tokenizer.decode(body_slice).strip()
        if not chunk_body:
            continue
        chunks.append(f"{header}{separator}{chunk_body}".strip())
        if start + available_body_tokens >= len(body_tokens):
            break
    return chunks, header


def build_chunk_records(
    row: dict,
    tokenizer,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    sanitized = sanitize_row(row)
    header_for_chunks = build_structured_header(row=sanitized, for_chunk=True)
    full_structured_text = build_structured_full_text(sanitized)
    legal_body = sanitized.get("გადაწყვეტილება", "")

    chunk_texts, header_used = _split_body_with_header(
        header_text=header_for_chunks,
        body_text=legal_body,
        tokenizer=tokenizer,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
    )

    if not chunk_texts:
        chunk_texts = [header_for_chunks]
        header_used = header_for_chunks

    records: list[dict] = []
    for idx, chunk_text in enumerate(chunk_texts):
        seed = f"{sanitized.get('ბრძანება', '')}:{idx}:{chunk_text}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, hashlib.sha1(seed.encode("utf-8")).hexdigest()))

        metadata = dict(sanitized)
        metadata.update(
            {
                "structured_header": header_used,
                "structured_full_text": full_structured_text,
                "chunk_index": idx,
            }
        )
        records.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "metadata": metadata,
            }
        )
    return records


def deduplicate_sources(rows: Iterable[dict]) -> list[dict]:
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[dict] = []

    for row in rows:
        source = {
            "document": clean_text(row.get("დოკუმენტი", "")),
            "order_number": clean_text(row.get("ბრძანება", "")),
            "category": clean_text(row.get("კატეგორია", "")),
            "date": clean_text(row.get("მიღების თარიღი", "")),
            "url": clean_text(row.get("item_page_link", "")),
        }
        key = (
            source["document"],
            source["order_number"],
            source["category"],
            source["date"],
            source["url"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique
