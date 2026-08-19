import json
import logging
from typing import List, Dict
from arxiv_seeker.llm import LLMClient
from arxiv_seeker.agent.schemas import JudgedPaper
from arxiv_seeker.agent.prompts import JUDGE_SYSTEM_PROMPT
from arxiv_seeker.api_client import Paper

logger = logging.getLogger(__name__)

class RelevanceJudge:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def judge(self, user_query: str, candidate_papers: List[Paper]) -> List[JudgedPaper]:
        if not candidate_papers:
            return []

        # Формируем компактное представление для LLM
        candidate_list = []
        for p in candidate_papers[:20]:  # не более 20, чтобы не перегрузить контекст
            candidate_list.append({
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": p.abstract[:300],  # обрезаем для экономии токенов
            })

        user_prompt = f"User message: {user_query}\n\nCandidates:\n{json.dumps(candidate_list, ensure_ascii=False, indent=2)}"

        raw = self.llm.chat(JUDGE_SYSTEM_PROMPT, user_prompt, json_mode=True)
        try:
            cleaned = raw.strip().strip("```json").strip("```").strip()
            data = json.loads(cleaned)
            # Ожидаем список объектов с ключами arxiv_id, keep, reason, fit_for_level
            judged = []
            for item in data:
                judged.append(JudgedPaper(
                    arxiv_id=item["arxiv_id"],
                    keep=item.get("keep", False),
                    reason=item.get("reason", ""),
                    fit_for_level=item.get("fit_for_level", False),
                ))
            return judged
        except Exception as e:
            logger.warning("Judge parsing failed (%s), keeping all candidates", e)
            # Фолбэк: оставляем все кандидаты
            return [
                JudgedPaper(arxiv_id=p.arxiv_id, keep=True, reason="", fit_for_level=True)
                for p in candidate_papers
            ]