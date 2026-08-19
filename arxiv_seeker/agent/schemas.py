from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SearchIntent:
    topics: List[str]                 # исправленные, канонические темы: ["backpropagation", "gradient descent"]
    arxiv_queries: List[str]          # готовые к отправке в SearchOrchestrator строки
    categories: List[str]             # ["cs.LG", "cs.NE"] — агент сам подбирает, если юзер не указал
    user_level: str                   # "beginner" | "intermediate" | "advanced"
    is_explicit_request: bool         # True если юзер сам назвал конкретную тему, а не просит "что почитать"
    clarifying_question: Optional[str] = None  # если запрос совсем непонятен

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