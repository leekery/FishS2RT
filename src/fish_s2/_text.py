"""Optional text segmentation that never rewrites user content."""

from __future__ import annotations


def split_sentences_exact(text: str) -> list[str]:
    """Split at clear sentence boundaries while preserving every character."""

    if not text:
        return []
    result: list[str] = []
    start = 0
    index = 0
    in_tag = False
    length = len(text)
    while index < length:
        if text.startswith("<|", index):
            in_tag = True
            index += 2
            continue
        if in_tag:
            if text.startswith("|>", index):
                in_tag = False
                index += 2
            else:
                index += 1
            continue
        char = text[index]
        is_decimal_dot = (
            char == "."
            and index > 0
            and index + 1 < length
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        )
        if char in ".!?…" and not is_decimal_dot:
            end = index + 1
            while end < length and text[end] in ".!?…\"'»”)]}":
                end += 1
            if end == length or text[end].isspace():
                while end < length and text[end].isspace():
                    end += 1
                segment = text[start:end]
                if segment:
                    result.append(segment)
                start = end
                index = end
                continue
        index += 1
    if start < length:
        result.append(text[start:])
    return result or [text]
