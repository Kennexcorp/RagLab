"""
Generator module for RAG system.
Uses LangChain's ChatPromptTemplate and provider-specific LLM wrappers
to replace manual prompt construction and per-provider API calls.
"""

from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import Config
from utils import setup_logging, timer


_HUMAN_MSG = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer: Provide a clear, concise answer based on the context above. "
    "If you reference specific data, mention the source."
)


class Generator:
    """Generate answers using LLM with retrieved context."""

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        api_key: str = None,
        temperature: float = None,
        system_prompt: str = None,
        log_level: str = "INFO",
    ):
        """
        Initialize Generator.

        Args:
            provider: LLM provider ('openai', 'anthropic', or 'ollama')
            model: Model name
            api_key: API key for the provider
            temperature: LLM temperature (0.0 = precise, 1.0 = creative)
            system_prompt: System message override
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

        self.provider = provider or Config.LLM_PROVIDER
        self.model = model or Config.LLM_MODEL
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        self.max_tokens = Config.LLM_MAX_TOKENS
        self.api_key = api_key or Config.get_llm_api_key()

        self.system_prompt = system_prompt or Config.SYSTEM_PROMPT
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", _HUMAN_MSG),
        ])
        self._llm = self._build_llm()
        self.chain = self._prompt | self._llm

        self.logger.info(
            f"Generator initialized: provider={self.provider}, model={self.model}"
        )

    def _build_llm(self):
        """Instantiate the LangChain LLM wrapper for the configured provider."""
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            if not self.api_key:
                raise ValueError("OpenAI API key is required")
            return ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
            )

        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            if not self.api_key:
                raise ValueError("Anthropic API key is required")
            return ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
            )

        elif self.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=self.model, temperature=self.temperature)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    @timer
    def generate(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer for a query using retrieved context.

        Args:
            query:                User query
            context:              Retrieved context (formatted text)
            conversation_history: Prior turns as [{"role": "user"|"assistant", "content": str}, ...]

        Returns:
            Response dictionary with answer and metadata
        """
        self.logger.info(f"Generating answer for query: '{query}'")

        if conversation_history:
            history_msgs = []
            for turn in conversation_history:
                if turn.get("role") == "user":
                    history_msgs.append(HumanMessage(content=turn["content"]))
                elif turn.get("role") == "assistant":
                    history_msgs.append(AIMessage(content=turn["content"]))
            assembled_human = _HUMAN_MSG.format(context=context, question=query)
            prompt_sent = (
                f"[System]\n{self.system_prompt}\n\n"
                + "".join(
                    f"[{'User' if isinstance(m, HumanMessage) else 'Assistant'}]\n{m.content}\n\n"
                    for m in history_msgs
                )
                + f"[User]\n{assembled_human}"
            )
            messages = (
                [SystemMessage(content=self.system_prompt)]
                + history_msgs
                + [HumanMessage(content=assembled_human)]
            )
            response = self._llm.invoke(messages)
        else:
            prompt_sent = (
                f"[System]\n{self.system_prompt}\n\n"
                f"[User]\n{_HUMAN_MSG.format(context=context, question=query)}"
            )
            response = self.chain.invoke({"context": context, "question": query})

        usage = getattr(response, "usage_metadata", None) or {}
        finish_reason = (
            response.response_metadata.get("finish_reason")
            if hasattr(response, "response_metadata")
            else None
        )

        self.logger.info("Answer generated successfully")
        return {
            "answer": response.content,
            "model": self.model,
            "tokens_used": usage.get("total_tokens"),
            "finish_reason": finish_reason,
            "prompt_sent": prompt_sent,
        }

    def generate_with_sources(
        self,
        query: str,
        context_data: Dict[str, Any],
        conversation_history: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer and include source citations.

        Args:
            query: User query
            context_data: Context data dict with 'context' and 'sources' keys

        Returns:
            Response with answer, citations, and sources
        """
        context = context_data.get("context", "")
        sources = context_data.get("sources", [])

        response = self.generate(query, context, conversation_history=conversation_history)
        response["sources"] = sources
        response["num_sources"] = len(sources)

        answer = response["answer"]
        if sources:
            citations = "\n\nSources:"
            for i, source in enumerate(sources, 1):
                source_info = source.get("metadata", {}).get("source", "Unknown")
                citations += f"\n[{i}] {source_info}"
            response["answer_with_citations"] = answer + citations
        else:
            response["answer_with_citations"] = answer

        return response


if __name__ == "__main__":
    import os

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: No LLM API key set. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env.")
    else:
        generator = Generator()

        context = """[Source 1: finance_report]
Q4 revenue increased by 25% compared to last year. Total revenue reached $10M.

[Source 2: analytics_dashboard]
Daily active users reached 1M milestone. User retention improved by 15%."""

        query = "How is our revenue performing?"
        response = generator.generate(query, context)

        print(f"\nQuery: {query}")
        print(f"\nAnswer: {response['answer']}")
        print(f"\nModel: {response['model']}")
        print(f"Tokens used: {response['tokens_used']}")
