from dataclasses import dataclass, field
from typing import List, Optional
@dataclass
class SearchIntent:
    topics: List[str]
    arxiv_queries: List[str]
    categories: List[str]
    domain: str                        # NEW — short field label, inferred, not hardcoded
    user_level: str
    is_explicit_request: bool
    clarifying_question: Optional[str] = None

@dataclass
class JudgedPaper:
    arxiv_id: str
    keep: bool
    reason: str        # 1 фраза, почему подходит/не подходит именно под query
    fit_for_level: bool

@dataclass
class AgentSearchResult:
    reply_text: str
    papers: List["Paper"]     # твой существующий Paper из api_client, отфильтрованный
    reasons: dict              # arxiv_id -> reason