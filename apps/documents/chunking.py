DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 200


def chunk_text(text: str, *, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_paragraph(paragraph, max_chars, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def _split_long_paragraph(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    step = max(max_chars - overlap, 1)
    return [paragraph[i:i + max_chars] for i in range(0, len(paragraph), step)]
