"""
Query expansion module for improving semantic search results
Expands short queries into related terms, synonyms, and variations before embedding
"""
import os
import logging
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands search queries to improve semantic search results using AI.
    
    Uses lightweight LLM (GPT-3.5-turbo) to intelligently expand queries
    with contextually relevant synonyms, related terms, and variations.
    """
    
    def __init__(self, use_llm: bool = True, llm_api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize query expander
        
        Args:
            use_llm: Whether to use LLM-based expansion (default: True)
            llm_api_key: API key for OpenAI (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use (default: "gpt-3.5-turbo" for lightweight/cheap)
        """
        self.use_llm = use_llm
        self.model = model
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
    
    def expand_query(self, query: str, max_expansions: int = 10) -> str:
        """
        Expand a query into related terms using AI
        
        Args:
            query: Original search query
            max_expansions: Maximum number of expansion terms to add
            
        Returns:
            Expanded query string with original + related terms
        """
        if not query or not query.strip():
            return query
        
        # Use LLM-based expansion if enabled and API key available
        if self.use_llm and self.llm_api_key:
            expanded_terms = self._llm_based_expansion(query, max_expansions)
            if expanded_terms and len(expanded_terms) > 1:
                # expanded_terms[0] is the original query, [1:] are the expansions
                # Combine original query with AI-generated expansions (skip first term which is the original)
                expansion_only = expanded_terms[1:] if expanded_terms[0].lower() == query.lower() else expanded_terms
                expanded_query = f"{query} {' '.join(expansion_only)}"
                logger.info(f"AI expanded query '{query}' to: {expanded_query[:200]}...")
                return expanded_query
            else:
                # LLM failed or returned only original
                logger.warning(f"LLM expansion failed for '{query}', using original query")
                return query
        else:
            # No LLM configured — use original query
            if not self.llm_api_key:
                logger.warning("No OpenAI API key found. Set OPENAI_API_KEY for AI expansion.")
            return query
    
    def get_expansion_phrases(self, query: str, max_phrases: int = 5) -> Tuple[List[str], bool]:
        """
        Return a list of distinct phrases (original + expansions) for multi-query retrieval.
        Use with Reciprocal Rank Fusion (RRF): embed each phrase separately, search each,
        then fuse result lists. Prefer this over concatenating into one string and embedding
        once, or over averaging embeddings (which dilutes signal).
        
        Args:
            query: Original search query
            max_phrases: Maximum number of phrases (original + expansion phrases)
            
        Returns:
            (phrases, expansion_unavailable): phrases from LLM expansion, or [query] when
            expansion is off or fails. expansion_unavailable is True when we wanted expansion
            but only have the single original query (no rule-based fallbacks).
        """
        if not query or not query.strip():
            return ([query] if query else [], False)
        
        qnorm = query.strip().lower()
        
        # LLM-only expansion: when off or fails, return single query only (no rule-based fallbacks)
        if not (self.use_llm and self.llm_api_key):
            return ([query.strip()], True)
        
        terms = self._llm_based_expansion(query, max_expansions=max_phrases)
        if not terms or (len(terms) == 1 and terms[0].lower().strip() == qnorm):
            return ([query.strip()], True)
        
        # Dedupe and cap phrases
        seen = set()
        phrases = []
        for t in terms:
            tl = t.lower().strip()
            if tl and tl not in seen:
                seen.add(tl)
                phrases.append(t.strip())
            if len(phrases) >= max_phrases:
                break
        if not phrases or phrases[0].lower() != query.lower().strip():
            phrases = [query.strip()] + [p for p in phrases if p.lower() != query.lower().strip()]
        return (phrases[:max_phrases], False)
    
    def _llm_based_expansion(self, query: str, max_expansions: int) -> List[str]:
        """
        LLM-based query expansion using OpenAI.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expansions
            
        Returns:
            List of expansion terms (includes original query as first term)
        """
        if not self.llm_api_key:
            return [query]
        try:
            return self._openai_expansion(query, max_expansions)
        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}. Using original query.")
        return [query]
    
    def _openai_expansion(self, query: str, max_expansions: int) -> List[str]:
        """Expand query using OpenAI API (lightweight GPT-3.5-turbo)"""
        try:
            import openai
            api_key = self.llm_api_key
            if not api_key:
                return []
            
            client = openai.OpenAI(api_key=api_key)
            
            # Prompt: preserve anchor terms/phrases so expansions stay focused (e.g. "budget" in "budget limitations", "mushroom" in topping queries)
            prompt = f"""You expand search queries for a qualitative corpus (e.g. interviews/surveys). Your job is to generate related search phrases that stay tightly focused on the same main concept(s) as the query and use wording respondents might actually say.

Given the user query: "{query}"

STEP 1 – Identify focus terms:
- Find the 1–3 key noun or noun-phrase anchors that carry the meaning.
  Examples: "budget limitations" → anchor phrase "budget limitations" (do NOT treat "limitations" alone as the anchor). "mushroom" → anchor term "mushroom". "who likes pepperoni?" → anchor term "pepperoni".
- Ignore generic words like "who", "what", "why", "when", "how", "limitations", "issues", "things" unless they are clearly the core concept.

STEP 2 – Generate expansions:
- At least 70% of the expansions MUST literally include the anchor term or phrase.
  - For "budget limitations": use variants like "budget limitations", "limited budget", "tight budget", "budget constraints", "budget issues". Avoid standalone "limitations" with no mention of budget.
  - For "mushroom": use variants like "mushrooms", "extra mushrooms"—not generic "toppings" or "favorite toppings" that drop the word "mushroom".
  - For "who likes pepperoni?": focus on pepperoni—e.g. "pepperoni lovers", "people who like pepperoni", "favorite pepperoni topping", "pepperoni preference".
- Only rarely include a broader category that omits the anchor (e.g. "pizza toppings" from "pepperoni"), and only if obviously useful.
- Prefer short, natural phrases (2–6 words) that someone might say in an interview.

STEP 3 – Format:
- Do NOT rephrase the whole query as a long sentence. Do NOT explain. Return ONLY a comma-separated list of phrases, most relevant first.

Terms:"""
            
            system_content = (
                "You expand search queries for retrieval. Identify the key anchor term or phrase in each query and ensure most of your expansion phrases literally include that anchor. Do not split compound concepts (e.g. keep 'budget' tied to 'budget limitations'); do not replace specific terms (e.g. 'mushroom') with generic ones (e.g. 'toppings'). Output only a comma-separated list of short phrases."
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,  # Enough for ~10-15 terms
                temperature=0.4,  # Balanced creativity/consistency
                timeout=10  # Fast timeout for responsiveness
            )
            
            expansions_text = response.choices[0].message.content.strip()
            # Clean up the response - remove any leading/trailing text
            expansions_text = expansions_text.split(":")[-1].strip() if ":" in expansions_text else expansions_text
            
            # Parse comma-separated terms
            expansions = [e.strip() for e in expansions_text.split(",") if e.strip()]
            
            # Filter out the original query if it appears in expansions (avoid duplicates)
            query_lower = query.lower()
            filtered_expansions = [
                exp for exp in expansions 
                if exp.lower() != query_lower and exp.lower() not in query_lower.split()
            ]
            
            # Limit to max_expansions
            filtered_expansions = filtered_expansions[:max_expansions]
            
            # Return list with original query first, then expansions
            result = [query]
            result.extend(filtered_expansions)
            
            return result
        except ImportError:
            logger.warning("OpenAI library not installed. Install with: pip install openai")
            return [query]  # Return original query only
        except Exception as e:
            logger.warning(f"OpenAI expansion failed: {e}")
            return [query]  # Return original query only
    
# Default instance for easy import
_default_expander = None

def get_query_expander(use_llm: bool = False, llm_api_key: Optional[str] = None) -> QueryExpander:
    """Get or create default query expander instance"""
    global _default_expander
    if _default_expander is None:
        _default_expander = QueryExpander(use_llm=use_llm, llm_api_key=llm_api_key)
    return _default_expander
