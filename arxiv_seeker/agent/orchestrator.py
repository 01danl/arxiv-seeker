import logging
from typing import List, Dict, Optional

from arxiv_seeker.agent.community_discovery import CommunityDiscoveryAgent
from arxiv_seeker.search import SearchOrchestrator
from arxiv_seeker.agent.intent import IntentAgent
from arxiv_seeker.agent.relevance_judge import RelevanceJudge
from arxiv_seeker.agent.schemas import AgentSearchResult
from arxiv_seeker.agent.seed_catalog import match_catalog, SeedPaper
from arxiv_seeker.api_client import Paper, ArxivClient

logger = logging.getLogger(__name__)


class AgentSearchOrchestrator:
    def __init__(self, config_overrides: dict | None = None):
        self.intent_agent = IntentAgent()
        self.judge = RelevanceJudge()
        self.search_orchestrator = SearchOrchestrator()
        self.config = config_overrides or {}
        self.community_agent = CommunityDiscoveryAgent()
        self._arxiv_client = ArxivClient()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def run(self, user_message: str) -> AgentSearchResult:
        # Шаг 1: понимание запроса
        intent = self.intent_agent.parse(user_message)
        logger.info(
            "INTENT: domain=%r explicit=%s user_level=%s queries=%s",
            intent.domain,
            intent.is_explicit_request,
            intent.user_level,
            intent.arxiv_queries,
        )
        if intent.clarifying_question:
            return AgentSearchResult(
                reply_text=f"Уточните, пожалуйста: {intent.clarifying_question}",
                papers=[],
                reasons={},
            )

        # Шаг 2: для неявных/новичковых запросов — ЗАРАНЕЕ подгружаем seed catalog.
        # Это гарантирует, что даже при слабом judging или плохих поисковых выдачах
        # пользователь получит проверенные foundational papers.
        seed_papers: List[Paper] = []
        if not intent.is_explicit_request:
            seed_papers = self._seed_catalog_fallback(intent.domain, set())
            logger.info("Seed catalog preloaded %d papers for domain=%r", len(seed_papers), intent.domain)

        # Шаг 3: собираем кандидатов (community discovery + arXiv/Tavily search)
        all_papers, seen_ids = self._gather_candidates(user_message, intent)

        # Вливаем seed papers в общий пул (они гарантированно качественные)
        for sp in seed_papers:
            if sp.arxiv_id not in seen_ids:
                seen_ids.add(sp.arxiv_id)
                all_papers.append(sp)

        if not all_papers:
            return AgentSearchResult(
                reply_text="Не удалось найти статьи по вашему запросу.",
                papers=[],
                reasons={},
            )

        # Шаг 4: оценка релевантности (LLM-джадж)
        # Отделяем seed papers — они предварительно проверены, их не фильтруем
        seed_ids = {sp.arxiv_id for sp in seed_papers}
        non_seed = [p for p in all_papers if p.arxiv_id not in seed_ids]

        if non_seed:
            judged = self.judge.judge(user_message, non_seed, domain=intent.domain)
            kept_papers = self._filter_and_sort(judged, non_seed)
        else:
            kept_papers = []

        # Seed papers всегда включаются (они гарантированно релевантны)
        kept_papers = seed_papers + kept_papers

        # Шаг 5: если мало — итерация с уточнёнными запросами
        min_papers = self.config.get("min_papers", 2)
        if len(kept_papers) < min_papers:
            logger.info(
                "Only %d papers after judge (need %d), trying refinement iteration",
                len(kept_papers),
                min_papers,
            )
            refined = self._refinement_iteration(user_message, intent, seen_ids)
            if refined:
                kept_papers = kept_papers + refined
                seen = set()
                deduped = []
                for p in kept_papers:
                    if p.arxiv_id not in seen:
                        seen.add(p.arxiv_id)
                        deduped.append(p)
                kept_papers = deduped

        # Шаг 6: ограничиваем финальное количество
        top_n = self.config.get("final_top_n", 6)
        kept_papers = kept_papers[:top_n]

        # Шаг 7: формируем ответ
        reasons_dict: dict = {}
        # Reasons from the judge (for non-seed papers)
        if non_seed:
            reasons_dict.update({
                j.arxiv_id: j.reason
                for j in judged
                if j.arxiv_id in {p.arxiv_id for p in kept_papers}
            })
        # Reasons for seed papers
        for sp in seed_papers:
            if sp.arxiv_id in {p.arxiv_id for p in kept_papers}:
                reasons_dict.setdefault(sp.arxiv_id, "Foundational paper — curated recommendation")

        if kept_papers:
            reply = f"Нашёл для вас {len(kept_papers)} подходящих статей по запросу.\n"
            if intent.user_level == "beginner":
                reply += "Я подобрал в основном обзорные и вводные материалы.\n"
        else:
            reply = (
                "Не нашёл статей, которые точно соответствуют вашему запросу. "
                "Попробуйте уточнить тему или назвать конкретные методы/алгоритмы."
            )

        return AgentSearchResult(
            reply_text=reply,
            papers=kept_papers,
            reasons=reasons_dict,
        )

    # ------------------------------------------------------------------
    #  Internal steps
    # ------------------------------------------------------------------
    def _gather_candidates(
        self, user_message: str, intent
    ) -> tuple[List[Paper], set]:
        all_papers: List[Paper] = []
        seen_ids: set[str] = set()

        # 2a: community discovery (для неявных запросов)
        if not intent.is_explicit_request:
            logger.info(
                "Запрос неявный, используем community-discovery для поиска рекомендаций."
            )
            titles = self.community_agent.find_recommended_titles(
                user_message, domain=intent.domain
            )
            logger.info("Community discovery returned %d titles: %s", len(titles), titles)
            for title in titles:
                paper = self.search_orchestrator.resolve_title(title)
                if paper and paper.arxiv_id not in seen_ids:
                    seen_ids.add(paper.arxiv_id)
                    paper.final_score = paper.similarity_score = 1.0
                    all_papers.append(paper)

        # 2b: поиск по сгенерированным arXiv-запросам
        for query in intent.arxiv_queries:
            if self.search_orchestrator.settings.search_backend == "tavily":
                papers = self.search_orchestrator.search_via_tavily(
                    query,
                    max_results=self.config.get("candidates_per_query", 8),
                )
            else:
                papers = self.search_orchestrator.search(
                    query,
                    max_results=self.config.get("candidates_per_query", 8),
                    sort_by="relevance",
                    use_cache=True,
                    rerank=True,
                )
            logger.info("Query %r returned %d papers", query, len(papers))
            for p in papers:
                if p.arxiv_id not in seen_ids:
                    seen_ids.add(p.arxiv_id)
                    all_papers.append(p)

        return all_papers, seen_ids

    @staticmethod
    def _filter_and_sort(judged, all_papers: List[Paper]) -> List[Paper]:
        kept_ids = {j.arxiv_id for j in judged if j.keep}
        kept = [p for p in all_papers if p.arxiv_id in kept_ids]
        kept.sort(key=lambda p: getattr(p, "final_score", 0.0), reverse=True)
        logger.info(
            "Judge: %d candidates → %d kept (%.0f%%)",
            len(judged),
            len(kept),
            100 * len(kept) / max(len(judged), 1),
        )
        return kept

    def _seed_catalog_fallback(
        self, domain: str, seen_ids: set[str]
    ) -> List[Paper]:
        """Try to fetch papers from the curated seed catalog for *domain*."""
        entries = match_catalog(domain)
        if not entries:
            return []

        ids_to_fetch = [e.arxiv_id for e in entries if e.arxiv_id not in seen_ids]
        if not ids_to_fetch:
            return []

        logger.info("Seed catalog: fetching %d papers", len(ids_to_fetch))
        try:
            papers = self._arxiv_client.get_by_ids(ids_to_fetch)
        except Exception as exc:
            logger.warning("Seed catalog fetch failed: %s", exc)
            return []

        for p in papers:
            p.final_score = p.similarity_score = 0.95  # high-confidence seed
            seen_ids.add(p.arxiv_id)

        return papers

    def _refinement_iteration(
        self,
        user_message: str,
        intent,
        seen_ids: set[str],
    ) -> List[Paper]:
        """One extra round: ask the intent agent to refine its queries."""
        # Ask the intent agent to generate more specific queries
        from arxiv_seeker.agent.prompts import INTENT_SYSTEM_PROMPT

        refine_prompt = (
            f"{INTENT_SYSTEM_PROMPT}\n\n"
            f"The previous search for \"{user_message}\" returned too few good papers. "
            f"Please generate 3-5 DIFFERENT, more specific arxiv_queries — try synonyms, "
            f"broader/sibling topics, or alternative technical terms within the same domain "
            f"({intent.domain}). Still follow all the original rules."
        )

        try:
            raw = self.intent_agent.llm.chat(refine_prompt, user_message, json_mode=True)
            import json
            data = json.loads(raw.strip().strip("```json").strip("```"))
            refined_queries = data.get("arxiv_queries", [])
            logger.info("Refinement generated queries: %s", refined_queries)
        except Exception as exc:
            logger.warning("Refinement intent generation failed: %s", exc)
            return []

        papers: List[Paper] = []
        for query in refined_queries[:4]:
            if self.search_orchestrator.settings.search_backend == "tavily":
                batch = self.search_orchestrator.search_via_tavily(
                    query,
                    max_results=self.config.get("candidates_per_query", 6),
                )
            else:
                batch = self.search_orchestrator.search(
                    query,
                    max_results=self.config.get("candidates_per_query", 6),
                    sort_by="relevance",
                    use_cache=False,
                    rerank=True,
                )
            for p in batch:
                if p.arxiv_id not in seen_ids:
                    seen_ids.add(p.arxiv_id)
                    papers.append(p)

        if not papers:
            return []

        # Judge the refined candidates too
        judged = self.judge.judge(user_message, papers, domain=intent.domain)
        return self._filter_and_sort(judged, papers)