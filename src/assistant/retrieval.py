from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    title: str
    section: str
    source: str
    text: str
    sha256: str
    tokens: frozenset[str]


def _tokens(value: str) -> frozenset[str]:
    lowered = value.lower()
    ascii_tokens = re.findall(r"[a-z0-9_./+-]{2,}", lowered)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    chinese_tokens = list(chinese) + [chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))]
    return frozenset(ascii_tokens + chinese_tokens)


class KnowledgeIndex:
    """Dependency-free lexical retrieval over versioned local guidance."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.project = Path(__file__).resolve().parents[2]
        self.root = Path(root or self.project / "knowledge" / "assistant")
        self.chunks = self._load()

    def _load(self) -> list[KnowledgeChunk]:
        if not self.root.exists():
            return []
        chunks: list[KnowledgeChunk] = []
        for path in sorted(self.root.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            title = path.stem.replace("_", " ").title()
            current_section = title
            buffer: list[str] = []

            def flush() -> None:
                if not buffer:
                    return
                text = "\n".join(buffer).strip()
                buffer.clear()
                if not text:
                    return
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"kb-{len(chunks) + 1:03d}",
                        title=title,
                        section=current_section,
                        source=path.relative_to(self.project).as_posix(),
                        text=text,
                        sha256=digest,
                        tokens=_tokens(f"{title} {current_section} {text}"),
                    )
                )

            for line in content.splitlines():
                if line.startswith("#"):
                    flush()
                    current_section = line.lstrip("#").strip() or title
                    if line.startswith("# "):
                        title = current_section
                    continue
                if sum(len(item) for item in buffer) + len(line) > 1400:
                    flush()
                buffer.append(line)
            flush()
        return chunks

    def search(self, query: str, *, limit: int = 4, language: str = "zh") -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self.chunks:
            overlap = query_tokens & chunk.tokens
            if not overlap:
                continue
            score = len(overlap) / max(1, len(query_tokens))
            phrase_bonus = 0.2 if query.lower() in chunk.text.lower() else 0.0
            scored.append((score + phrase_bonus, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        result = []
        for score, chunk in scored[: max(1, min(limit, 8))]:
            excerpt = re.sub(r"\s+", " ", chunk.text).strip()
            if len(excerpt) > 700:
                excerpt = excerpt[:697].rstrip() + "..."
            result.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "source": chunk.source,
                    "excerpt": excerpt,
                    "score": round(score, 4),
                    "sha256": chunk.sha256,
                    "language": language,
                }
            )
        return result
