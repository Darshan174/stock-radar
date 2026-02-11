"""
Stock Radar - RAG Validator

RAGAS-style validation for RAG-enhanced stock analyses.
Validates that AI responses are properly grounded in retrieved context.

Metrics:
- Faithfulness: Is the answer grounded in retrieved context?
- Context Relevancy: Are retrieved documents relevant to the query?
- Groundedness: Are claims supported by source data?
- Temporal Validity: Is context recent enough for stock analysis?
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

try:
    from litellm import completion
except ImportError:
    completion = None

logger = logging.getLogger(__name__)


@dataclass
class RAGValidationResult:
    """Container for RAG validation metrics."""
    
    # Core metrics (0-100 scale)
    faithfulness_score: float = 0.0        # Answer grounded in context
    context_relevancy_score: float = 0.0   # Retrieved docs are relevant
    groundedness_score: float = 0.0        # Claims supported by data
    temporal_validity_score: float = 0.0   # Context recency
    
    # Overall score (weighted average)
    overall_score: float = 0.0
    quality_grade: str = "Unknown"  # A, B, C, D, F
    
    # Detailed breakdown
    validation_details: Dict[str, Any] = field(default_factory=dict)
    claims_verified: int = 0
    claims_total: int = 0
    sources_used: int = 0
    oldest_source_age_hours: float = 0.0
    
    # Metadata
    validation_time_ms: int = 0
    validator_model: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API."""
        return {
            "faithfulness_score": round(self.faithfulness_score, 1),
            "context_relevancy_score": round(self.context_relevancy_score, 1),
            "groundedness_score": round(self.groundedness_score, 1),
            "temporal_validity_score": round(self.temporal_validity_score, 1),
            "overall_score": round(self.overall_score, 1),
            "quality_grade": self.quality_grade,
            "claims_verified": self.claims_verified,
            "claims_total": self.claims_total,
            "sources_used": self.sources_used,
            "oldest_source_age_hours": round(self.oldest_source_age_hours, 1),
            "validation_time_ms": self.validation_time_ms,
            "validation_details": self.validation_details,
        }


class RAGValidator:
    """
    RAGAS-style validator for RAG-enhanced stock analyses.
    
    Validates:
    1. Faithfulness - Does the answer only use information from context?
    2. Context Relevancy - Are retrieved docs actually relevant?
    3. Groundedness - Can each claim be traced to source data?
    4. Temporal Validity - Is the context fresh enough?
    """
    
    # Weights for overall score calculation
    WEIGHTS = {
        "faithfulness": 0.35,
        "context_relevancy": 0.25,
        "groundedness": 0.25,
        "temporal_validity": 0.15,
    }
    
    # Thresholds for intraday vs longterm (hours)
    RECENCY_THRESHOLDS = {
        "intraday": 24,    # 24 hours for intraday
        "longterm": 168,   # 7 days for longterm
    }
    
    def __init__(self, llm_model: str = "zai/glm-4.7"):
        """
        Initialize RAG validator.
        
        Args:
            llm_model: LLM to use for validation checks
        """
        self.llm_model = llm_model
        logger.info(f"RAGValidator initialized with model: {llm_model}")
    
    def validate_analysis(
        self,
        query: str,
        answer: str,
        retrieved_context: List[Dict[str, Any]],
        analysis_mode: str = "intraday",
        source_data: Dict[str, Any] = None
    ) -> RAGValidationResult:
        """
        Validate a RAG-enhanced analysis.
        
        Args:
            query: Original query/context description
            answer: LLM-generated analysis/response
            retrieved_context: List of retrieved documents with metadata
            analysis_mode: 'intraday' or 'longterm'
            source_data: Original source data (quote, indicators, etc.)
            
        Returns:
            RAGValidationResult with all metrics
        """
        start_time = time.time()
        result = RAGValidationResult()
        
        try:
            # Extract context text
            context_texts = self._extract_context_texts(retrieved_context)
            context_combined = "\n\n".join(context_texts)
            
            # Calculate each metric
            result.faithfulness_score = self._calculate_faithfulness(
                answer, context_combined, source_data
            )
            
            result.context_relevancy_score = self._calculate_context_relevancy(
                query, retrieved_context
            )
            
            result.groundedness_score, claims_info = self._calculate_groundedness(
                answer, context_combined, source_data
            )
            result.claims_verified = claims_info.get("verified", 0)
            result.claims_total = claims_info.get("total", 0)
            
            result.temporal_validity_score = self._calculate_temporal_validity(
                retrieved_context, analysis_mode
            )
            
            # Calculate overall score
            result.overall_score = (
                self.WEIGHTS["faithfulness"] * result.faithfulness_score +
                self.WEIGHTS["context_relevancy"] * result.context_relevancy_score +
                self.WEIGHTS["groundedness"] * result.groundedness_score +
                self.WEIGHTS["temporal_validity"] * result.temporal_validity_score
            )
            
            # Assign quality grade
            result.quality_grade = self._get_quality_grade(result.overall_score)
            
            # Additional metadata
            result.sources_used = len(retrieved_context)
            result.oldest_source_age_hours = self._get_oldest_source_age(retrieved_context)
            result.validation_time_ms = int((time.time() - start_time) * 1000)
            result.validator_model = self.llm_model
            
            # Store detailed breakdown
            result.validation_details = {
                "faithfulness": {
                    "score": result.faithfulness_score,
                    "description": "Answer uses information from retrieved context"
                },
                "context_relevancy": {
                    "score": result.context_relevancy_score,
                    "description": "Retrieved documents are relevant to query"
                },
                "groundedness": {
                    "score": result.groundedness_score,
                    "claims_verified": result.claims_verified,
                    "claims_total": result.claims_total,
                    "description": "Claims can be traced to source data"
                },
                "temporal_validity": {
                    "score": result.temporal_validity_score,
                    "oldest_source_hours": result.oldest_source_age_hours,
                    "threshold_hours": self.RECENCY_THRESHOLDS.get(analysis_mode, 24),
                    "description": "Context is recent enough for analysis type"
                }
            }
            
            logger.info(
                f"RAG validation complete: overall={result.overall_score:.1f}, "
                f"grade={result.quality_grade}, time={result.validation_time_ms}ms"
            )
            
        except Exception as e:
            logger.error(f"RAG validation failed: {e}")
            result.validation_details = {"error": str(e)}
        
        return result
    
    def _extract_context_texts(self, retrieved_context: List[Dict[str, Any]]) -> List[str]:
        """Extract text content from retrieved documents."""
        texts = []
        for doc in retrieved_context:
            # Handle different document structures
            text = (
                doc.get("content") or 
                doc.get("reasoning") or 
                doc.get("summary") or 
                doc.get("headline") or
                doc.get("reason") or
                str(doc)
            )
            if text:
                texts.append(str(text))
        return texts
    
    def _calculate_faithfulness(
        self,
        answer: str,
        context: str,
        source_data: Dict[str, Any] = None
    ) -> float:
        """
        Calculate faithfulness score - is the answer grounded in context?
        
        Uses LLM to check if answer statements are supported by context.
        """
        if not answer or not context:
            return 50.0  # Neutral score if no data
        
        if not completion:
            return self._calculate_faithfulness_heuristic(answer, context)
        
        prompt = f"""Evaluate if this stock analysis is faithful to the provided context.

CONTEXT (Retrieved Information):
{context[:2000]}

ANALYSIS TO VERIFY:
{answer[:1000]}

Score the faithfulness from 0-100:
- 100: All claims directly supported by context
- 75: Most claims supported, minor extrapolations
- 50: Mixed - some supported, some not
- 25: Few claims supported by context  
- 0: Analysis contradicts or ignores context

Respond with JSON only:
{{"score": number, "reasoning": "brief explanation"}}"""

        try:
            response = completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            
            import json
            result = self._extract_json(response.choices[0].message.content)
            return float(result.get("score", 50))
            
        except Exception as e:
            logger.warning(f"LLM faithfulness check failed: {e}")
            return self._calculate_faithfulness_heuristic(answer, context)
    
    def _calculate_faithfulness_heuristic(self, answer: str, context: str) -> float:
        """Heuristic faithfulness check when LLM unavailable."""
        # Check keyword overlap between answer and context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "can", "need", "dare", "ought", "used", "to", "of", "in",
                     "for", "on", "with", "at", "by", "from", "as", "into",
                     "through", "during", "before", "after", "above", "below",
                     "between", "under", "again", "further", "then", "once"}
        
        answer_words = answer_words - stopwords
        context_words = context_words - stopwords
        
        if not answer_words:
            return 50.0
        
        overlap = len(answer_words & context_words)
        overlap_ratio = overlap / len(answer_words)
        
        return min(100, overlap_ratio * 120)  # Scale up slightly
    
    def _calculate_context_relevancy(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate context relevancy - are retrieved docs relevant to query?
        
        Uses similarity scores from retrieval if available.
        """
        if not retrieved_context:
            return 0.0
        
        # Use existing similarity scores from retrieval
        similarities = []
        for doc in retrieved_context:
            sim = doc.get("similarity", doc.get("score", 0.5))
            if isinstance(sim, (int, float)):
                similarities.append(float(sim))
        
        if not similarities:
            return 50.0  # Default neutral score
        
        # Weight by position (first results more important)
        weighted_scores = []
        for i, sim in enumerate(similarities):
            weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, 0.25...
            weighted_scores.append(sim * weight * 100)
        
        # Calculate weighted average
        total_weight = sum(1.0 / (i + 1) for i in range(len(similarities)))
        weighted_avg = sum(weighted_scores) / total_weight if total_weight > 0 else 50
        
        return min(100, max(0, weighted_avg))
    
    def _calculate_groundedness(
        self,
        answer: str,
        context: str,
        source_data: Dict[str, Any] = None
    ) -> tuple[float, Dict]:
        """
        Calculate groundedness - can claims be traced to source?
        
        Checks if specific numbers/facts in answer match source data.
        """
        claims_info = {"verified": 0, "total": 0, "details": []}
        
        if not answer:
            return 50.0, claims_info
        
        # Extract numerical claims from answer
        import re
        numbers_in_answer = re.findall(r'\d+\.?\d*%?', answer)
        claims_info["total"] = len(numbers_in_answer)
        
        if not numbers_in_answer:
            return 75.0, claims_info  # No numerical claims to verify
        
        # Check if numbers appear in context or source data
        context_str = context or ""
        source_str = str(source_data) if source_data else ""
        combined_source = context_str + " " + source_str
        
        verified = 0
        for num in numbers_in_answer:
            if num in combined_source:
                verified += 1
                claims_info["details"].append({"claim": num, "verified": True})
            else:
                claims_info["details"].append({"claim": num, "verified": False})
        
        claims_info["verified"] = verified
        
        if claims_info["total"] == 0:
            return 75.0, claims_info
        
        score = (verified / claims_info["total"]) * 100
        return score, claims_info
    
    def _calculate_temporal_validity(
        self,
        retrieved_context: List[Dict[str, Any]],
        analysis_mode: str
    ) -> float:
        """
        Calculate temporal validity - is context recent enough?
        
        Intraday needs very fresh data, longterm can be older.
        """
        if not retrieved_context:
            return 50.0  # No context to validate
        
        threshold_hours = self.RECENCY_THRESHOLDS.get(analysis_mode, 24)
        now = datetime.now(timezone.utc)
        
        ages_hours = []
        for doc in retrieved_context:
            # Try to get timestamp
            timestamp = (
                doc.get("created_at") or 
                doc.get("timestamp") or 
                doc.get("published_at")
            )
            
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        # Handle ISO format
                        if timestamp.endswith('Z'):
                            timestamp = timestamp[:-1] + '+00:00'
                        doc_time = datetime.fromisoformat(timestamp)
                    else:
                        doc_time = timestamp
                    
                    if doc_time.tzinfo is None:
                        doc_time = doc_time.replace(tzinfo=timezone.utc)
                    
                    age_hours = (now - doc_time).total_seconds() / 3600
                    ages_hours.append(age_hours)
                except Exception:
                    pass
        
        if not ages_hours:
            return 70.0  # Can't determine, assume somewhat recent
        
        # Score based on oldest document (weakest link)
        oldest_age = max(ages_hours)
        
        if oldest_age <= threshold_hours:
            # Fresh enough - linear scale from 100 to 70
            score = 100 - (oldest_age / threshold_hours) * 30
        else:
            # Too old - linear decay from 70 to 0
            excess = oldest_age - threshold_hours
            score = max(0, 70 - (excess / threshold_hours) * 70)
        
        return score
    
    def _get_oldest_source_age(self, retrieved_context: List[Dict[str, Any]]) -> float:
        """Get age of oldest source in hours."""
        now = datetime.now(timezone.utc)
        oldest = 0.0
        
        for doc in retrieved_context:
            timestamp = doc.get("created_at") or doc.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        if timestamp.endswith('Z'):
                            timestamp = timestamp[:-1] + '+00:00'
                        doc_time = datetime.fromisoformat(timestamp)
                    else:
                        doc_time = timestamp
                    
                    if doc_time.tzinfo is None:
                        doc_time = doc_time.replace(tzinfo=timezone.utc)
                    
                    age = (now - doc_time).total_seconds() / 3600
                    oldest = max(oldest, age)
                except Exception:
                    pass
        
        return oldest
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        import json
        try:
            return json.loads(text)
        except:
            pass
        
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        
        return {}


# =============================================================================
# Convenience Functions
# =============================================================================

_validator_instance: Optional[RAGValidator] = None


def get_validator() -> RAGValidator:
    """Get singleton RAGValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = RAGValidator()
    return _validator_instance


def validate_rag_analysis(
    query: str,
    answer: str,
    retrieved_context: List[Dict[str, Any]],
    analysis_mode: str = "intraday",
    source_data: Dict[str, Any] = None
) -> RAGValidationResult:
    """
    Convenience function to validate a RAG analysis.
    
    Args:
        query: Original query
        answer: Generated answer
        retrieved_context: Retrieved documents
        analysis_mode: 'intraday' or 'longterm'
        source_data: Source data for grounding check
        
    Returns:
        RAGValidationResult
    """
    return get_validator().validate_analysis(
        query=query,
        answer=answer,
        retrieved_context=retrieved_context,
        analysis_mode=analysis_mode,
        source_data=source_data
    )


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing RAG Validator")
    print("=" * 50)
    
    validator = RAGValidator()
    
    # Test data
    test_query = "RELIANCE.NS intraday analysis with RSI 65 MACD bullish"
    test_answer = """
    Based on the technical indicators, RELIANCE shows bullish momentum. 
    RSI at 65 indicates strength without being overbought. 
    MACD crossover suggests potential upside. 
    Target price: 2900, Stop loss: 2800.
    """
    test_context = [
        {
            "content": "Previous RELIANCE analysis showed RSI at 62, signal was BUY.",
            "similarity": 0.85,
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "content": "MACD bullish crossover patterns historically led to 2-3% gains.",
            "similarity": 0.78,
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        }
    ]
    test_source = {
        "price": 2847.50,
        "rsi_14": 65,
        "macd": 15.5
    }
    
    result = validator.validate_analysis(
        query=test_query,
        answer=test_answer,
        retrieved_context=test_context,
        analysis_mode="intraday",
        source_data=test_source
    )
    
    print(f"\nValidation Results:")
    print(f"  Overall Score: {result.overall_score:.1f}")
    print(f"  Quality Grade: {result.quality_grade}")
    print(f"\n  Breakdown:")
    print(f"    Faithfulness: {result.faithfulness_score:.1f}")
    print(f"    Context Relevancy: {result.context_relevancy_score:.1f}")
    print(f"    Groundedness: {result.groundedness_score:.1f} ({result.claims_verified}/{result.claims_total} claims verified)")
    print(f"    Temporal Validity: {result.temporal_validity_score:.1f}")
    print(f"\n  Metadata:")
    print(f"    Sources Used: {result.sources_used}")
    print(f"    Oldest Source: {result.oldest_source_age_hours:.1f} hours")
    print(f"    Validation Time: {result.validation_time_ms}ms")
    
    print("\n" + "=" * 50)
    print("RAG Validator tests completed")
