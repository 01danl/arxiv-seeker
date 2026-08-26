import json
import logging
from typing import List, Dict
from arxiv_seeker.llm import LLMClient
from arxiv_seeker.agent.schemas import JudgedPaper
from arxiv_seeker.agent.prompts import JUDGE_SYSTEM_PROMPT_TEMPLATE
from arxiv_seeker.api_client import Paper

logger = logging.getLogger(__name__)


class RelevanceJudge:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def judge(
        self,
        user_query: str,
        candidate_papers: List[Paper],
        domain: str,
        max_candidates: int = 20,
    ) -> List[JudgedPaper]:
        if not candidate_papers:
            return []

        # Формируем компактное представление для LLM
        candidate_list = []
        for p in candidate_papers[:max_candidates]:
            candidate_list.append({
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": p.abstract[:300],
            })

        user_prompt = (
            f"User message: {user_query}\n\n"
            f"Candidates:\n{json.dumps(candidate_list, ensure_ascii=False, indent=2)}"
        )

        system_prompt = JUDGE_SYSTEM_PROMPT_TEMPLATE.format(domain=domain)
        raw = self.llm.chat(system_prompt, user_prompt, json_mode=True)

        try:
            return self._parse_judge_response(raw, candidate_papers)
        except Exception as exc:
            logger.warning(
                "Judge parsing failed (%s), raw=%r — retrying without json_mode",
                exc,
                raw[:500],
            )
            # One retry without json_mode (some providers/models handle plain
            # text better with structured prompts)
            try:
                raw2 = self.llm.chat(system_prompt, user_prompt, json_mode=False)
                return self._parse_judge_response(raw2, candidate_papers)
            except Exception as exc2:
                logger.error(
                    "Judge retry also failed (%s). Falling back to SBERT top-N only.",
                    exc2,
                )
                return self._fallback(user_query, candidate_papers)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_judge_response(raw: str, candidate_papers: List[Paper]) -> List[JudgedPaper]:
        """Parse the LLM JSON response.  Raises on unrecoverable parse errors."""
        cleaned = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)

        # json_mode enforces a top-level object; unwrap it.
        if isinstance(data, dict):
            data = (
                data.get("papers")
                or next((v for v in data.values() if isinstance(v, list)), None)
            )
            if data is None:
                raise ValueError(
                    f"Expected a 'papers' key or list value, got keys: {list(data.keys())}"
                )

        if not isinstance(data, list):
            raise TypeError(f"Expected a list after unwrapping, got {type(data).__name__}")

        judged: List[JudgedPaper] = []
        seen_ids = set()
        for item in data:
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict judge item: %r", item)
                continue
            try:
                aid = item["arxiv_id"]
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                judged.append(JudgedPaper(
                    arxiv_id=aid,
                    keep=item.get("keep", False),
                    reason=item.get("reason", ""),
                    fit_for_level=item.get("fit_for_level", False),
                ))
            except KeyError as ke:
                logger.warning("Judge item missing required key %s: %r", ke, item)

        if not judged:
            raise ValueError("No valid JudgedPaper items parsed from response")

        return judged

    @staticmethod
    def _fallback(user_query: str, candidate_papers: List[Paper]) -> List[JudgedPaper]:
        """Last-resort fallback: keep the top papers by SBERT similarity only.

        This is intentionally conservative — it keeps at most 3 papers so we
        don't flood the user with noise when the judge is broken.
        """
        logger.warning(
            "Judge fallback: keeping top-3 by existing similarity_score for %d candidates",
            len(candidate_papers),
        )
        sorted_papers = sorted(
            candidate_papers, key=lambda p: getattr(p, "similarity_score", 0.0), reverse=True
        )
        return [
            JudgedPaper(
                arxiv_id=p.arxiv_id,
                keep=True,
                reason="(fallback — judge unavailable)",
                fit_for_level=True,
            )
            for p in sorted_papers[:3]
        ]