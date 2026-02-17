"""
Generator module for RAG system.
Handles LLM integration and answer generation using retrieved context.
"""

import logging
from typing import Dict, Any, Optional, List
import openai
from anthropic import Anthropic

from config import Config
from utils import setup_logging, timer, count_tokens


class Generator:
    """Generate answers using LLM with retrieved context."""

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        api_key: str = None,
        log_level: str = "INFO",
    ):
        """
        Initialize Generator.

        Args:
            provider: LLM provider (openai, anthropic, ollama)
            model: Model name
            api_key: API key for the provider
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

        self.provider = provider or Config.LLM_PROVIDER
        self.model = model or Config.LLM_MODEL
        self.temperature = Config.LLM_TEMPERATURE
        self.max_tokens = Config.LLM_MAX_TOKENS

        # Set up API key
        self.api_key = api_key or Config.get_llm_api_key()

        # Initialize client based on provider
        if self.provider == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key is required")
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            self.logger.info(f"Initialized OpenAI client with model: {self.model}")

        elif self.provider == "anthropic":
            if not self.api_key:
                raise ValueError("Anthropic API key is required")
            self.client = Anthropic(api_key=self.api_key)
            self.logger.info(f"Initialized Anthropic client with model: {self.model}")

        elif self.provider == "ollama":
            # Ollama runs locally, no API key needed
            self.logger.info(f"Using Ollama with model: {self.model}")
            self.logger.warning(
                "Ollama support is basic - install ollama package for full support"
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt_template = """You are a helpful assistant that answers questions about organization dashboard data.

Use the following context to answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer: Provide a clear, concise answer based on the context above. If you reference specific data, mention the source."""

        prompt = prompt_template.format(context=context, query=query)
        return prompt

    @timer
    def generate_with_openai(self, prompt: str) -> Dict[str, Any]:
        """
        Generate answer using OpenAI API.

        Args:
            prompt: Formatted prompt

        Returns:
            Response dictionary with answer and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions about organization dashboard data.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "model": self.model,
                "tokens_used": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason,
            }

        except Exception as e:
            self.logger.error(f"Error generating with OpenAI: {str(e)}")
            raise

    @timer
    def generate_with_anthropic(self, prompt: str) -> Dict[str, Any]:
        """
        Generate answer using Anthropic API.

        Args:
            prompt: Formatted prompt

        Returns:
            Response dictionary with answer and metadata
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response.content[0].text

            return {
                "answer": answer,
                "model": self.model,
                "tokens_used": response.usage.input_tokens
                + response.usage.output_tokens,
                "finish_reason": response.stop_reason,
            }

        except Exception as e:
            self.logger.error(f"Error generating with Anthropic: {str(e)}")
            raise

    @timer
    def generate_with_ollama(self, prompt: str) -> Dict[str, Any]:
        """
        Generate answer using Ollama (local).

        Args:
            prompt: Formatted prompt

        Returns:
            Response dictionary with answer and metadata
        """
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "answer": result.get("response", ""),
                    "model": self.model,
                    "tokens_used": None,  # Ollama doesn't always return token counts
                    "finish_reason": "stop",
                }
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Error generating with Ollama: {str(e)}")
            raise

    def generate(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate answer for a query using retrieved context.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Response dictionary with answer and metadata
        """
        self.logger.info(f"Generating answer for query: '{query}'")

        # Create prompt
        prompt = self.create_prompt(query, context)

        # Log prompt tokens
        prompt_tokens = count_tokens(prompt)
        self.logger.debug(f"Prompt tokens: {prompt_tokens}")

        # Generate based on provider
        if self.provider == "openai":
            response = self.generate_with_openai(prompt)
        elif self.provider == "anthropic":
            response = self.generate_with_anthropic(prompt)
        elif self.provider == "ollama":
            response = self.generate_with_ollama(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.logger.info("Answer generated successfully")
        return response

    def generate_with_sources(
        self, query: str, context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate answer and include source citations.

        Args:
            query: User query
            context_data: Context data with sources

        Returns:
            Response with answer and sources
        """
        context = context_data.get("context", "")
        sources = context_data.get("sources", [])

        # Generate answer
        response = self.generate(query, context)

        # Add sources to response
        response["sources"] = sources
        response["num_sources"] = len(sources)

        # Create formatted answer with citations
        answer = response["answer"]
        if sources:
            citations = "\n\nSources:"
            for i, source in enumerate(sources, 1):
                metadata = source.get("metadata", {})
                source_info = metadata.get("source", "Unknown")
                citations += f"\n[{i}] {source_info}"

            response["answer_with_citations"] = answer + citations
        else:
            response["answer_with_citations"] = answer

        return response


if __name__ == "__main__":
    # Example usage (requires API key)
    import os

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Set it in .env file to test.")
        print("Example usage shown below:")
        print(
            """
        generator = Generator(provider="openai", model="gpt-3.5-turbo")
        
        context = "Q4 revenue increased by 25% to $10M. User engagement is up 15%."
        query = "How is our revenue performing?"
        
        response = generator.generate(query, context)
        print(response['answer'])
        """
        )
    else:
        generator = Generator(provider="openai", model="gpt-3.5-turbo")

        # Sample context and query
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
