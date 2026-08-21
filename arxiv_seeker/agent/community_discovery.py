import json
import logging
from typing import List

from arxiv_seeker.llm import LLMClient
from arxiv_seeker.tavily_client import TavilyClient
from arxiv_seeker.agent.prompts import PAPER_EXTRACTION_SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class CommunityDiscoveryAgent:
    """Finds papers that people actually recommend/discuss on the web, for vague/exploratory queries."""

    def __init__(self, llm: LLMClient | None = None, tavily: TavilyClient | None = None):
        self.llm = llm or LLMClient()
        self.tavily = tavily or TavilyClient()

    def find_recommended_titles(self, user_query: str, domain: str, max_sources: int = 8) -> List[str]:
        search_query = f"{user_query} recommended papers reading list must read"
        sources = self.tavily.search_general(search_query, max_results=max_sources)
        if not sources:
            return []

        snippets = "\n\n".join(f"[{s['title']}]({s['url']})\n{s['content']}" for s in sources)
        system_prompt = PAPER_EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(domain=domain)
        raw = self.llm.chat(system_prompt, snippets, json_mode=True)

        try:
            cleaned = raw.strip().strip("```json").strip("```").strip()
            titles = json.loads(cleaned)
            return [t for t in titles if isinstance(t, str)][:10]
        except Exception as e:
            logger.warning("Paper title extraction failed (%s)", e)
            return []