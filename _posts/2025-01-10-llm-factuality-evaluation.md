---
title: 'Revolutionizing LLM Evaluation: A 71% Cycle-Time Reduction with FactScore and LangGraph 📊🔍'
date: 2025-01-10
permalink: /posts/llm-factuality-evaluation/
tags:
  - LLM
  - FactScore
  - LangGraph
  - NER
  - Evaluation
  - Meta
  - Patent-Pending
---

In the rapidly evolving world of Large Language Models (LLMs), one of the biggest challenges is evaluating the factual accuracy of generated content. At Honeywell, we faced this exact problem: how do you systematically evaluate the factuality of LLM outputs across thousands of documents while maintaining speed, accuracy, and scalability? The solution we developed not only solved this problem but achieved a remarkable **71% reduction in cycle-time** (from 2 weeks to just 4 days) and has a **patent pending**. Let me walk you through this fascinating journey.

## The Problem: Unreliable LLM Evaluation ⚠️

Large Language Models are incredibly powerful, but they have a notorious problem: they can generate content that sounds convincing but is factually incorrect. In enterprise environments like Honeywell, where accuracy is paramount, we needed a robust system to:

- **Automatically evaluate** the factual accuracy of LLM-generated content
- **Scale to thousands** of documents and reports
- **Provide deterministic** and reproducible results
- **Integrate seamlessly** with existing workflows

Traditional evaluation methods were either too slow (manual review), too unreliable (simple similarity metrics), or too domain-specific (rule-based systems). We needed something revolutionary.

## Enter FactScore: Meta's Game-Changing Framework 🧠

[FactScore](https://github.com/shmsw25/FActScore), developed by researchers at Meta, provided the foundation for our solution. FactScore evaluates the factual accuracy of text by:

1. **Decomposing** complex statements into atomic facts
2. **Verifying** each atomic fact against reliable knowledge sources
3. **Aggregating** individual fact scores into an overall factuality score

However, FactScore alone wasn't enough for our enterprise needs. We needed to adapt and scale it significantly.

## Our Innovation: Deterministic Pipelines with LangGraph 🔄

The breakthrough came when we integrated FactScore with **LangGraph** - a framework for building stateful, multi-actor applications with LLMs. This combination allowed us to create deterministic evaluation pipelines that could process complex workflows reliably.

### Architecture Overview

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation
import asyncio
from typing import TypedDict, List, Dict

class EvaluationState(TypedDict):
    document_text: str
    atomic_facts: List[str]
    fact_scores: Dict[str, float]
    ner_entities: List[Dict]
    final_score: float
    confidence: float

class FactualityEvaluator:
    def __init__(self):
        self.workflow = self._build_evaluation_graph()
        
    def _build_evaluation_graph(self) -> StateGraph:
        """Build the deterministic evaluation pipeline"""
        workflow = StateGraph(EvaluationState)
        
        # Add nodes for each evaluation step
        workflow.add_node("extract_facts", self.extract_atomic_facts)
        workflow.add_node("ner_analysis", self.perform_ner_analysis)
        workflow.add_node("fact_verification", self.verify_facts)
        workflow.add_node("score_aggregation", self.aggregate_scores)
        workflow.add_node("confidence_estimation", self.estimate_confidence)
        
        # Define the evaluation flow
        workflow.set_entry_point("extract_facts")
        workflow.add_edge("extract_facts", "ner_analysis")
        workflow.add_edge("ner_analysis", "fact_verification")
        workflow.add_edge("fact_verification", "score_aggregation")
        workflow.add_edge("score_aggregation", "confidence_estimation")
        workflow.add_edge("confidence_estimation", END)
        
        return workflow.compile()
```

### Atomic Fact Extraction

The first step in our pipeline involves breaking down complex statements into verifiable atomic facts:

```python
import spacy
from transformers import pipeline

class AtomicFactExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.fact_extractor = pipeline(
            "text2text-generation",
            model="facebook/bart-large-cnn"
        )
    
    def extract_atomic_facts(self, state: EvaluationState) -> EvaluationState:
        """Extract atomic facts from complex text"""
        text = state["document_text"]
        
        # Use dependency parsing to identify claim structures
        doc = self.nlp(text)
        claims = []
        
        for sent in doc.sents:
            # Extract subject-predicate-object triplets
            triplets = self._extract_triplets(sent)
            claims.extend(triplets)
        
        # Further decompose complex claims
        atomic_facts = []
        for claim in claims:
            decomposed = self._decompose_claim(claim)
            atomic_facts.extend(decomposed)
        
        state["atomic_facts"] = atomic_facts
        return state
    
    def _extract_triplets(self, sentence):
        """Extract subject-predicate-object triplets from sentence"""
        triplets = []
        for token in sentence:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token.text
                predicate = token.head.text
                
                # Find object
                for child in token.head.children:
                    if child.dep_ in ["dobj", "attr", "prep"]:
                        obj = child.text
                        triplets.append(f"{subject} {predicate} {obj}")
        
        return triplets
```

## Enhanced NER Analysis for Entity Verification 🎯

Named Entity Recognition (NER) plays a crucial role in our evaluation system. We developed an enhanced NER pipeline that goes beyond traditional entity extraction:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import requests

class EnhancedNERAnalyzer:
    def __init__(self):
        self.ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        self.entity_verifier = EntityVerifier()
    
    def perform_ner_analysis(self, state: EvaluationState) -> EvaluationState:
        """Perform enhanced NER analysis with verification"""
        text = state["document_text"]
        
        # Extract entities
        entities = self.ner_pipeline(text)
        
        # Enhance entities with additional context
        enhanced_entities = []
        for entity in entities:
            enhanced = self._enhance_entity(entity, text)
            verification_result = self.entity_verifier.verify_entity(enhanced)
            enhanced["verification"] = verification_result
            enhanced_entities.append(enhanced)
        
        state["ner_entities"] = enhanced_entities
        return state
    
    def _enhance_entity(self, entity: dict, full_text: str) -> dict:
        """Add context and confidence to entity"""
        # Add surrounding context
        start, end = entity["start"], entity["end"]
        context_start = max(0, start - 50)
        context_end = min(len(full_text), end + 50)
        
        entity["context"] = full_text[context_start:context_end]
        entity["confidence_enhanced"] = self._calculate_confidence(entity)
        
        return entity

class EntityVerifier:
    def __init__(self):
        self.knowledge_base = self._initialize_kb()
    
    def verify_entity(self, entity: dict) -> dict:
        """Verify entity against knowledge base"""
        entity_text = entity["word"]
        entity_type = entity["entity_group"]
        
        verification = {
            "exists": self._check_existence(entity_text, entity_type),
            "accuracy_score": self._calculate_accuracy(entity_text, entity_type),
            "sources": self._find_sources(entity_text)
        }
        
        return verification
```

## The Deterministic Pipeline Magic ✨

The key innovation of our system is its deterministic nature. Unlike traditional ML evaluation methods that can produce different results on identical inputs, our pipeline ensures consistent, reproducible results:

```python
class DeterministicFactVerifier:
    def __init__(self):
        self.knowledge_sources = [
            WikipediaKB(),
            InternalKB(),
            IndustryStandardsKB()
        ]
        self.verification_cache = {}
    
    def verify_facts(self, state: EvaluationState) -> EvaluationState:
        """Deterministically verify atomic facts"""
        atomic_facts = state["atomic_facts"]
        ner_entities = state["ner_entities"]
        
        fact_scores = {}
        
        for fact in atomic_facts:
            # Generate deterministic hash for caching
            fact_hash = self._generate_fact_hash(fact)
            
            if fact_hash in self.verification_cache:
                score = self.verification_cache[fact_hash]
            else:
                score = self._verify_single_fact(fact, ner_entities)
                self.verification_cache[fact_hash] = score
            
            fact_scores[fact] = score
        
        state["fact_scores"] = fact_scores
        return state
    
    def _verify_single_fact(self, fact: str, entities: List[Dict]) -> float:
        """Verify a single atomic fact against knowledge sources"""
        verification_scores = []
        
        for kb in self.knowledge_sources:
            score = kb.verify_fact(fact, entities)
            verification_scores.append(score)
        
        # Weighted average based on source reliability
        weights = [0.4, 0.4, 0.2]  # Wikipedia, Internal, Standards
        final_score = sum(s * w for s, w in zip(verification_scores, weights))
        
        return final_score
```

## Results: 71% Cycle-Time Reduction 📈

The impact of our system has been transformative:

### Before Our System:
- **Manual Review Time**: 2 weeks per batch of documents
- **Inconsistent Results**: Human reviewers had 60-70% agreement
- **Limited Scalability**: Could only process 50-100 documents per cycle
- **High Cost**: Required multiple senior engineers for review

### After Implementation:
- **Automated Processing**: 4 days per batch (71% reduction!)
- **Consistent Results**: 95%+ reproducibility across runs
- **Massive Scalability**: Process 1000+ documents per cycle
- **Cost Effective**: Minimal human intervention required

### Key Performance Metrics:

```python
class PerformanceMetrics:
    def __init__(self):
        self.metrics = {
            "cycle_time_reduction": 0.71,  # 71% reduction
            "throughput_increase": 10.0,    # 10x more documents
            "accuracy_improvement": 0.25,   # 25% better accuracy
            "cost_reduction": 0.85,         # 85% cost reduction
            "consistency_score": 0.95       # 95% reproducible results
        }
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        return f"""
        Performance Report - LLM Factuality Evaluation System
        =====================================================
        
        Cycle Time: {(1-self.metrics['cycle_time_reduction'])*100:.0f}% of original
        Throughput: {self.metrics['throughput_increase']:.1f}x increase
        Accuracy: +{self.metrics['accuracy_improvement']*100:.0f}% improvement
        Cost: {(1-self.metrics['cost_reduction'])*100:.0f}% of original
        Consistency: {self.metrics['consistency_score']*100:.0f}% reproducible
        """
```

## Patent-Pending Innovation 🏆

Our approach has been so innovative that we've filed for a patent covering:

1. **Novel combination** of FactScore with LangGraph for deterministic evaluation
2. **Enhanced NER integration** for entity-aware fact verification
3. **Scalable caching mechanism** for reproducible results
4. **Multi-source knowledge verification** with weighted confidence scoring

The patent application covers the unique architectural approach that enables both high accuracy and deterministic behavior in LLM evaluation systems.

## Technical Deep Dive: Implementation Challenges 🔧

### Challenge 1: Handling Ambiguous Facts

```python
class AmbiguityResolver:
    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.disambiguation_model = DisambiguationModel()
    
    def resolve_ambiguous_fact(self, fact: str, context: str) -> List[str]:
        """Resolve ambiguous facts into specific, verifiable claims"""
        ambiguity_score = self.context_analyzer.detect_ambiguity(fact)
        
        if ambiguity_score > 0.7:
            # High ambiguity - need disambiguation
            resolved_facts = self.disambiguation_model.disambiguate(
                fact, context
            )
            return resolved_facts
        else:
            return [fact]
```

### Challenge 2: Scaling Knowledge Base Queries

```python
class ScalableKnowledgeBase:
    def __init__(self):
        self.vector_db = ChromaDB()
        self.cache_layer = RedisCache()
        self.query_optimizer = QueryOptimizer()
    
    def batch_verify_facts(self, facts: List[str]) -> Dict[str, float]:
        """Efficiently verify multiple facts in parallel"""
        # Group similar queries
        query_groups = self.query_optimizer.group_queries(facts)
        
        results = {}
        for group in query_groups:
            batch_results = self._verify_fact_batch(group)
            results.update(batch_results)
        
        return results
```

## Future Enhancements and Roadmap 🔮

Our system continues to evolve. Here's what's coming next:

1. **Real-time Evaluation**: Streaming fact verification as content is generated
2. **Domain-Specific Knowledge Bases**: Specialized verification for different industries
3. **Confidence Calibration**: Better uncertainty quantification for borderline cases
4. **Multi-language Support**: Extending to non-English content evaluation

## Conclusion: The Future of LLM Evaluation 🚀

Building this LLM factuality evaluation system has been one of the most rewarding projects of my career. The combination of cutting-edge research (FactScore), innovative frameworks (LangGraph), and practical engineering has created something truly special.

The 71% cycle-time reduction isn't just a number - it represents a fundamental shift in how we can reliably evaluate AI-generated content at scale. As LLMs become more prevalent in enterprise applications, robust evaluation systems like this will be essential for maintaining trust and accuracy.

The patent-pending nature of our approach validates the novelty and potential impact of this work. I'm excited to see how this technology will shape the future of AI evaluation and contribute to more reliable AI systems across industries.

---

**Interested in learning more about LLM evaluation or discussing factuality assessment?** Feel free to connect with me at [yashpathania704@gmail.com](mailto:yashpathania704@gmail.com) or on [LinkedIn](https://linkedin.com/in/yashhere). I'd love to discuss how these techniques might apply to your use cases!

*Next up: I'll be sharing insights about building the Common Order Taking (COT) system at Tata 1mg and how we achieved a 30% throughput boost with microservices architecture.*
