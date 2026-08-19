import json, logging
from arxiv_seeker.llm import LLMClient
from arxiv_seeker.agent.schemas import SearchIntent
from arxiv_seeker.agent.prompts import INTENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class IntentAgent:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def parse(self, user_message: str) -> SearchIntent:
        raw = self.llm.chat(INTENT_SYSTEM_PROMPT, user_message, json_mode=True)
        try:
            data = json.loads(raw.strip().strip("```json").strip("```"))
            return SearchIntent(**data)
        except Exception as e:
            logger.warning("Intent parse failed (%s), falling back to raw query", e)
            # graceful fallback: treat the raw message as a single query
            return SearchIntent(
                topics=[user_message], arxiv_queries=[user_message],
                categories=[], user_level="intermediate",
                is_explicit_request=True, clarifying_question=None,
            )