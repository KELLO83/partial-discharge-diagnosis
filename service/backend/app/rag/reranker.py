from __future__ import annotations

from dataclasses import dataclass

from service.backend.app.rag.documents import RagSearchHit
from service.backend.app.rag.query_constraints import RagQueryConstraints, hit_matches_constraints
from service.backend.app.schemas import MetadataInput


BASE_RELEVANCE_WEIGHT = 0.72
LABEL_MATCH_BONUS = 0.30
LABEL_MISMATCH_PENALTY = 0.12
SENSOR_MATCH_BONUS = 0.06
EQUIPMENT_TYPE_MATCH_BONUS = 0.04
INSULATOR_MATCH_BONUS = 0.04
QUERY_MATCH_BONUS = 0.24
QUERY_TITLE_MATCH_BONUS = 0.12
CONSTRAINT_MATCH_BONUS = 0.42
DIVERSITY_MIN_SCORE = 0.35
QUERY_TOKEN_SUFFIXES = (
    "이란",
    "란",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
)
SOURCE_PRIOR_BONUS = {
    "rulebook": 0.03,
    "sop": 0.02,
    "dataset_case": 0.01,
}


@dataclass(frozen=True, slots=True)
class RerankedHit:
    hit: RagSearchHit
    score: float


def rerank_hits(
    hits: list[RagSearchHit],
    metadata: MetadataInput | None,
    candidate_label_ids: tuple[int, ...],
    top_k: int,
    query_text: str | None = None,
    constraints: RagQueryConstraints | None = None,
) -> list[RerankedHit]:
    if top_k <= 0:
        return []
    ranked_pairs = sorted(
        (
            (_score_hit(hit, metadata, candidate_label_ids, query_text, constraints), index, hit)
            for index, hit in enumerate(hits)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    selected = _seed_diverse_sources(ranked_pairs, top_k)
    selected_keys = {hit.chunk_key for _, _, hit in selected}
    for pair in ranked_pairs:
        if len(selected) >= top_k:
            break
        if pair[2].chunk_key not in selected_keys:
            selected.append(pair)
            selected_keys.add(pair[2].chunk_key)
    selected.sort(key=lambda item: (-item[0], item[1]))
    return [RerankedHit(hit=hit, score=round(score, 4)) for score, _, hit in selected[:top_k]]


def _seed_diverse_sources(
    ranked_pairs: list[tuple[float, int, RagSearchHit]],
    top_k: int,
) -> list[tuple[float, int, RagSearchHit]]:
    if top_k < 4:
        return []
    selected: list[tuple[float, int, RagSearchHit]] = []
    for source_type in ("rulebook", "sop"):
        best = next(
            (
                pair
                for pair in ranked_pairs
                if pair[2].source_type == source_type and pair[0] >= DIVERSITY_MIN_SCORE
            ),
            None,
        )
        if best is not None:
            selected.append(best)
    return selected


def _score_hit(
    hit: RagSearchHit,
    metadata: MetadataInput | None,
    candidate_label_ids: tuple[int, ...],
    query_text: str | None,
    constraints: RagQueryConstraints | None,
) -> float:
    score = hit.relevance * BASE_RELEVANCE_WEIGHT
    score += SOURCE_PRIOR_BONUS.get(hit.source_type, 0.0)
    score += _label_score(hit, candidate_label_ids)
    score += _metadata_score(hit, metadata)
    score += _query_text_score(hit, query_text)
    score += _constraint_score(hit, constraints)
    return max(0.0, min(score, 1.0))


def _label_score(hit: RagSearchHit, candidate_label_ids: tuple[int, ...]) -> float:
    if not candidate_label_ids or hit.label_id is None:
        return 0.0
    if hit.label_id in candidate_label_ids:
        return LABEL_MATCH_BONUS
    return -LABEL_MISMATCH_PENALTY


def _metadata_score(hit: RagSearchHit, metadata: MetadataInput | None) -> float:
    if metadata is None:
        return 0.0
    score = 0.0
    if _same_text(hit.sensor_type, metadata.sensor_type):
        score += SENSOR_MATCH_BONUS
    if _same_text(hit.equipment_type, metadata.equipment_type):
        score += EQUIPMENT_TYPE_MATCH_BONUS
    if _same_text(hit.insulator_type, metadata.insulator_type or metadata.insulator_name):
        score += INSULATOR_MATCH_BONUS
    return score


def _query_text_score(hit: RagSearchHit, query_text: str | None) -> float:
    tokens = _query_tokens(query_text)
    if not tokens:
        return 0.0
    title = _compact_text(hit.title)
    haystack = _compact_text(" ".join([hit.title, hit.text, hit.source]))
    title_matches = sum(1 for token in tokens if token in title)
    matched_tokens = sum(1 for token in tokens if token in haystack)
    if matched_tokens == 0:
        return 0.0
    return (
        QUERY_MATCH_BONUS * (matched_tokens / len(tokens))
        + QUERY_TITLE_MATCH_BONUS * (title_matches / len(tokens))
    )


def _constraint_score(hit: RagSearchHit, constraints: RagQueryConstraints | None) -> float:
    if constraints is None or not constraints.has_constraints:
        return 0.0
    return CONSTRAINT_MATCH_BONUS if hit_matches_constraints(hit, constraints) else 0.0


def _same_text(left: str | None, right: str | None) -> bool:
    return _clean_text(left) != "" and _clean_text(left) == _clean_text(right)


def _clean_text(value: str | None) -> str:
    return "" if value is None else value.strip().lower()


def _query_tokens(value: str | None) -> list[str]:
    if value is None:
        return []
    return [
        token
        for token in (_normalize_query_token(_compact_text(part)) for part in value.split())
        if len(token) >= 2
    ]


def _normalize_query_token(value: str) -> str:
    for suffix in QUERY_TOKEN_SUFFIXES:
        if value.endswith(suffix) and len(value) > len(suffix) + 1:
            return value[: -len(suffix)]
    return value


def _compact_text(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())
