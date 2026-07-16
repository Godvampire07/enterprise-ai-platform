"""
Prompt Builder Service.

Constructs the system prompt and user prompt for the RAG pipeline.
The prompt is structured to:
  1. Ground the LLM strictly in retrieved context
  2. Prevent hallucination with explicit instructions
  3. Require the model to say "I don't know" when context is insufficient
  4. Include source attribution for traceability

Position in the RAG pipeline:
  RetrieverService → **PromptService** → LLMProvider
"""

from dataclasses import dataclass
from typing import List

from backend.app.core.logging import logger
from backend.app.services.retriever_service import RetrievedChunk


@dataclass
class ConstructedPrompt:
    """The two-part prompt ready for the LLM.

    Attributes:
        system_prompt: Instructions defining the model's role and constraints.
        user_prompt: The context + question assembled for the model.
    """

    system_prompt: str
    user_prompt: str


# Constant system prompt — the core grounding instructions.
# Separated from the class so it can be tested and reviewed independently.
SYSTEM_PROMPT = """You are an expert AI assistant for an Enterprise Document Management Platform.

Your ONLY source of truth is the CONTEXT provided below. Follow these rules strictly:

1. Answer ONLY based on the information found in the provided CONTEXT sections.
2. NEVER use prior knowledge, training data, or make assumptions beyond the context.
3. If the context does not contain sufficient information to answer the question, respond with:
   "I don't have enough information in the provided documents to answer this question."
4. When referencing information, naturally mention which document and page it comes from.
5. Be precise, professional, and concise.
6. If the context contains conflicting information from different sources, acknowledge the discrepancy and present both perspectives with their sources.
7. Do NOT fabricate citations, page numbers, or document names."""


class PromptService:
    """Builds structured prompts from retrieved chunks and user questions."""

    def build_prompt(
        self,
        chunks: List[RetrievedChunk],
        question: str,
    ) -> ConstructedPrompt:
        """Assemble a grounded RAG prompt from retrieved context chunks.

        The context section numbers each chunk with its source attribution
        (document name, page number) so the LLM can reference them naturally.

        Args:
            chunks: Retrieved chunks with metadata from RetrieverService.
            question: The user's original question.

        Returns:
            ConstructedPrompt with system and user prompts ready for the LLM.
        """
        if not chunks:
            logger.info("PromptService building prompt with no context chunks")
            context_section = (
                "No relevant context was found in the indexed documents."
            )
        else:
            context_section = self._format_context(chunks)

        user_prompt = self._assemble_user_prompt(context_section, question)

        logger.debug(
            f"Prompt built | context_chunks={len(chunks)} | "
            f"prompt_length={len(user_prompt)} chars"
        )

        return ConstructedPrompt(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into numbered context blocks with source attribution.

        Each chunk is wrapped with clear delimiters and metadata so the LLM
        can distinguish between sources and cite them accurately.
        """
        context_parts: List[str] = []

        for i, chunk in enumerate(chunks, start=1):
            page_info = f"Page {chunk.page_number}" if chunk.page_number else "Page N/A"

            context_parts.append(
                f"[Source {i}] — {chunk.document_name} | {page_info} | "
                f"Chunk {chunk.chunk_index}\n"
                f"{chunk.text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _assemble_user_prompt(
        self, context_section: str, question: str
    ) -> str:
        """Combine the context and question into the final user prompt."""
        return (
            f"=== CONTEXT ===\n\n"
            f"{context_section}\n\n"
            f"=== END CONTEXT ===\n\n"
            f"Based ONLY on the context above, please answer the following question:\n\n"
            f"Question: {question}"
        )
