"""Answer generation layer for the Constitution RAG API.

This module owns the *generation* seam of the system. The baseline
``ExtractiveBaselineGenerator`` does NOT fabricate an LLM answer: it returns a
clearly-labelled answer built only from retrieved passages, so no
constitutional claim can be invented here.

Satish and Pooran can replace the generator later by implementing the same
interface (LLM grounded generation, LoRA/QLoRA fine-tuned model, etc.):

    class MyRagGeneration:
        def generate(self, question: str, passages: list[RetrievedPassage]) -> str: ...

and passing it to ``src.rag.create_orchestrator(generator=MyRagGeneration())``.
No API-layer changes are required to swap the generator.

Citation metadata is never decided by the generator; the orchestrator builds
citations from the retrieved chunk metadata only.
"""

from __future__ import annotations

from src.schemas import RetrievedPassage

ANSWER_BASELINE_NOTE = (
    "This is the extractive baseline: it returns the most relevant passages "
    "retrieved from the Constitution corpus without LLM rewriting. A grounded "
    "answer-generation model will replace this step."
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I could not find enough supporting text in the available Constitution "
    "corpus to answer reliably. Please try rephrasing the question or asking "
    "about a specific Article."
)


class GenerationService:
    """Base class / informal interface for answer generators.

    Expected contract:
    * Input: user ``question`` and ranked ``passages`` (may be empty).
    * Output: plain answer text (no HTML/Markdown markup guarantees).
    * Must not invent Article numbers, pages or URLs; the API resolves
      citations from retrieved metadata independently of this text.
    """

    name = "base"

    def generate(self, question: str, passages: list[RetrievedPassage]) -> str:
        raise NotImplementedError("Subclasses must implement generate().")


class ExtractiveBaselineGenerator(GenerationService):
    """Bridging generator used until the team's LLM integration is ready.

    It does not claim to be a language model. It surfaces the top retrieved
    passage(s) verbatim and is transparent about that behaviour.
    """

    name = "extractive_baseline"

    def generate(self, question: str, passages: list[RetrievedPassage]) -> str:
        if not passages:
            return INSUFFICIENT_EVIDENCE_ANSWER

        top = passages[0]
        excerpt = self._excerpt(top, limit=400)
        location = self._describe(top)
        lead = (
            f"Based on the retrieved Constitution text ({location}), the most "
            f"relevant passage is:\n\n\"{excerpt}\""
        )

        extra = [p for p in passages[1:3] if p.chunk.text]
        if extra:
            more = "\n\n".join(
                f"- ({self._describe(p)}) \"{self._excerpt(p, limit=200)}\"" for p in extra
            )
            lead += "\n\nOther closely related passages:\n" + more

        return lead + "\n\n" + ANSWER_BASELINE_NOTE

    @staticmethod
    def _describe(passage: RetrievedPassage) -> str:
        parts = []
        if passage.chunk.article:
            parts.append(str(passage.chunk.article))
        if passage.chunk.page is not None:
            parts.append(f"PDF page {passage.chunk.page}")
        return ", ".join(parts) if parts else "source passage"

    @staticmethod
    def _excerpt(passage: RetrievedPassage, limit: int) -> str:
        text = " ".join(passage.chunk.text.split())
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + "…"
