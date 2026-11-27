"""Reviewable and reversible near-duplicate candidate handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
import os
from pathlib import Path

from documind.chunking.base import Chunk
from documind.vectorstore.chroma import ChromaVectorStore


class DuplicateStatus(StrEnum):
    """Human-review state; changing the status never deletes source chunks."""

    REVIEW = "review"
    DUPLICATE = "duplicate"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    """A pair of similar chunks awaiting or having received review."""

    first_chunk_id: str
    second_chunk_id: str
    similarity: float
    status: DuplicateStatus = DuplicateStatus.REVIEW

    def __post_init__(self) -> None:
        if self.first_chunk_id >= self.second_chunk_id:
            raise ValueError("Candidate chunk IDs must be sorted and different")
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError("similarity must be between 0 and 1")


class DuplicateDetector:
    """Find high-cosine-similarity pairs already present in a Chroma index."""

    def __init__(self, threshold: float = 0.95, candidates_per_chunk: int = 10) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if candidates_per_chunk < 2:
            raise ValueError("candidates_per_chunk must be at least 2")
        self.threshold = threshold
        self.candidates_per_chunk = candidates_per_chunk

    def detect(
        self,
        chunks: list[Chunk],
        vector_store: ChromaVectorStore,
    ) -> list[DuplicateCandidate]:
        """Return candidate pairs without removing or hiding any indexed chunk."""
        indexed_ids = set(vector_store.list_chunk_ids())
        missing = {chunk.chunk_id for chunk in chunks} - indexed_ids
        if missing:
            raise ValueError("Index all chunks before duplicate detection")
        per_chunk_results = vector_store.query_many(
            [chunk.text for chunk in chunks],
            n_results=self.candidates_per_chunk,
        )
        pairs: dict[tuple[str, str], float] = {}
        for chunk, results in zip(chunks, per_chunk_results, strict=True):
            for result in results:
                other_id = result.chunk.chunk_id
                if other_id == chunk.chunk_id or result.score < self.threshold:
                    continue
                pair = tuple(sorted((chunk.chunk_id, other_id)))
                pairs[pair] = max(pairs.get(pair, 0.0), min(1.0, result.score))
        return [
            DuplicateCandidate(first, second, similarity)
            for (first, second), similarity in sorted(pairs.items())
        ]


class DuplicateRegistry:
    """Persist review decisions while keeping every original chunk intact."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def all(self) -> list[DuplicateCandidate]:
        """Load recorded pairs and review states from JSONL."""
        if not self.path.exists():
            return []
        records: list[DuplicateCandidate] = []
        with self.path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    records.append(
                        DuplicateCandidate(
                            first_chunk_id=value["first_chunk_id"],
                            second_chunk_id=value["second_chunk_id"],
                            similarity=float(value["similarity"]),
                            status=DuplicateStatus(value["status"]),
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid duplicate registry row {line_number} in {self.path}: {exc}"
                    ) from exc
        return sorted(records, key=lambda item: (item.first_chunk_id, item.second_chunk_id))

    def record_candidates(
        self,
        candidates: list[DuplicateCandidate],
    ) -> int:
        """Save detections while retaining existing human decisions."""
        records = {
            (item.first_chunk_id, item.second_chunk_id): item
            for item in self.all()
        }
        for candidate in candidates:
            key = (candidate.first_chunk_id, candidate.second_chunk_id)
            current = records.get(key)
            records[key] = DuplicateCandidate(
                candidate.first_chunk_id,
                candidate.second_chunk_id,
                candidate.similarity,
                current.status if current else DuplicateStatus.REVIEW,
            )
        self._replace(list(records.values()))
        return len(candidates)

    def set_status(
        self,
        first_chunk_id: str,
        second_chunk_id: str,
        status: DuplicateStatus,
    ) -> DuplicateCandidate:
        """Set or reverse a review status for a previously detected pair."""
        first, second = sorted((first_chunk_id, second_chunk_id))
        records = self.all()
        for index, record in enumerate(records):
            if (record.first_chunk_id, record.second_chunk_id) == (first, second):
                updated = DuplicateCandidate(first, second, record.similarity, status)
                records[index] = updated
                self._replace(records)
                return updated
        raise KeyError(f"No duplicate candidate recorded for {first!r} and {second!r}")

    def _replace(self, candidates: list[DuplicateCandidate]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                for candidate in sorted(
                    candidates,
                    key=lambda item: (item.first_chunk_id, item.second_chunk_id),
                ):
                    output.write(
                        json.dumps(
                            {**asdict(candidate), "status": candidate.status.value},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
