import logging
from typing import List, Dict
from arxiv_seeker.agent.community_discovery import CommunityDiscoveryAgent
from arxiv_seeker.search import SearchOrchestrator
from arxiv_seeker.agent.intent import IntentAgent
from arxiv_seeker.agent.relevance_judge import RelevanceJudge
from arxiv_seeker.agent.schemas import AgentSearchResult
from arxiv_seeker.api_client import Paper

logger = logging.getLogger(__name__)

class AgentSearchOrchestrator:
    def __init__(self, config_overrides: dict = None):
        self.intent_agent = IntentAgent()
        self.judge = RelevanceJudge()
        self.search_orchestrator = SearchOrchestrator()
        self.config = config_overrides or {}
        self.community_agent = CommunityDiscoveryAgent()

    def run(self, user_message: str) -> AgentSearchResult:
        # Шаг 1: понимание запроса
        intent = self.intent_agent.parse(user_message)
        if intent.clarifying_question:
            return AgentSearchResult(
                reply_text=f"Уточните, пожалуйста: {intent.clarifying_question}",
                papers=[],
                reasons={},
            )

        all_papers = []
        seen_ids = set()

        # Шаг 2: если запрос расплывчатый – ищем рекомендации сообществ
        if not intent.is_explicit_request:
            logger.info("Запрос неявный, используем community-discovery для поиска рекомендаций.")
            titles = self.community_agent.find_recommended_titles(user_message, domain=intent.domain)
            for title in titles:
                paper = self.search_orchestrator.resolve_title(title)
                if paper and paper.arxiv_id not in seen_ids:
                    seen_ids.add(paper.arxiv_id)
                    paper.final_score = paper.similarity_score = 1.0
                    all_papers.append(paper)

    # Шаг 3: поиск по сгенерированным arXiv-запросам (основной поток)
        for query in intent.arxiv_queries:
            if self.search_orchestrator.settings.search_backend == "tavily":
                papers = self.search_orchestrator.search_via_tavily(
                    query, max_results=self.config.get("candidates_per_query", 8)
                )
            else:
                papers = self.search_orchestrator.search(
                    query,
                    max_results=self.config.get("candidates_per_query", 8),
                    sort_by="relevance",
                    use_cache=True,
                    rerank=True,
                )
            for p in papers:
                if p.arxiv_id not in seen_ids:
                    seen_ids.add(p.arxiv_id)
                    all_papers.append(p)

        if not all_papers:
            return AgentSearchResult(
                reply_text="Не удалось найти статьи по вашему запросу.",
                papers=[],
                reasons={},
            )

        # Шаг 4: оценка релевантности (LLM-джадж)
        judged = self.judge.judge(user_message, all_papers, domain=intent.domain)

        reasons_dict = {j.arxiv_id: j.reason for j in judged}
        kept_ids = {j.arxiv_id for j in judged if j.keep}
        kept_papers = [p for p in all_papers if p.arxiv_id in kept_ids]

        # Ограничиваем финальное количество
        top_n = self.config.get("final_top_n", 6)
        kept_papers = kept_papers[:top_n]

        # Шаг 5: формируем ответ
        if kept_papers:
            reply = f"Нашёл для вас {len(kept_papers)} подходящих статей по запросу.\n"
            if intent.user_level == "beginner":
                reply += " Я подобрал в основном обзорные и вводные материалы.\n"
        else:
            reply = "Не нашёл статей, которые точно соответствуют вашему запросу. Попробуйте уточнить тему."

        return AgentSearchResult(
            reply_text=reply,
            papers=kept_papers,
            reasons=reasons_dict,
        )