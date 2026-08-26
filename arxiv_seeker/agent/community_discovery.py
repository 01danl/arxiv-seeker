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

    def find_recommended_titles(
        self,
        user_query: str,
        domain: str,
        max_sources: int = 10,
    ) -> List[str]:
        # Build a domain-aware search query that targets curated lists and
        # practitioner recommendations rather than generic definitions.
        search_query = (
            f'site:reddit.com OR site:github.com OR site:news.ycombinator.com '
            f'"{domain}" recommended papers must read foundational'
        )
        # Also try a broader query for diversity
        sources = self.tavily.search_general(
            search_query, max_results=max_sources, domain=domain,
        )
        if len(sources) < 4:
            # Supplementary broader search — "best papers", "reading list", etc.
            broad_query = (
                f'"{domain}" best papers reading list for beginners '
                f'must-read foundational important'
            )
            extra = self.tavily.search_general(
                broad_query, max_results=max_sources, domain=domain,
            )
            seen_urls = {s["url"] for s in sources}
            for s in extra:
                if s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    sources.append(s)

        if not sources:
            logger.warning("Community discovery returned no sources for domain=%r", domain)
            return []

        snippets = "\n\n".join(
            f"[{s['title']}]({s['url']})\n{s['content']}" for s in sources
        )
        system_prompt = PAPER_EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(domain=domain)
        raw = self.llm.chat(system_prompt, snippets, json_mode=True)

        try:
            return self._parse_titles(raw)
        except Exception as e:
            logger.warning("Paper title extraction failed (%s), raw=%r", e, raw[:500])
            return []

    @staticmethod
    def _parse_titles(raw: str) -> List[str]:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)

        if isinstance(data, dict):
            if "titles" in data:
                data = data["titles"]
            else:
                # Try the first list-valued key
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break
                else:
                    raise ValueError(f"No list found in response keys: {list(data.keys())}")

        if not isinstance(data, list):
            raise TypeError(f"Expected a list, got {type(data).__name__}")

        titles = [t for t in data if isinstance(t, str)][:10]
        logger.info("Community discovery extracted %d titles", len(titles))
        return titles