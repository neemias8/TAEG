"""
Evaluation utilities for the TAEG project.

This module wraps common evaluation metrics (ROUGE, BERTScore), provides a
simple temporal coherence proxy, and orchestrates comparisons between the TAEG
model and baseline systems.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as GeometricData

try:  # Optional visualisation dependencies
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_LIBS_AVAILABLE = True
except ImportError:  # pragma: no cover - plotting is optional
    PLOT_LIBS_AVAILABLE = False

try:  # Optional text metrics
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:  # pragma: no cover
    BERTSCORE_AVAILABLE = False

from .data_loader import DataLoader, Event
from .graph_builder import TAEGGraphBuilder
from .models import BaseModel, TAEGModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration block for evaluation experiments."""

    compute_rouge: bool = True
    compute_bertscore: bool = True
    compute_temporal_coherence: bool = True

    rouge_types: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    rouge_use_stemmer: bool = True

    bertscore_model: str = "distilbert-base-uncased"
    bertscore_lang: str = "en"

    save_results: bool = True
    save_predictions: bool = True
    output_dir: str = "outputs/evaluation"
    create_plots: bool = True
    plot_dir: str = "outputs/plots"


@dataclass
class ModelResults:
    """Container for evaluation artefacts of a single model."""

    model_name: str
    predictions: List[str]
    rouge_scores: Optional[Dict[str, float]] = None
    bertscore_scores: Optional[Dict[str, float]] = None
    temporal_coherence_score: Optional[float] = None
    inference_time: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class ROUGEEvaluator:
    """Wrapper around `rouge_score` for convenience."""

    def __init__(self, rouge_types: List[str], use_stemmer: bool = True) -> None:
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge_score is not installed")
        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=use_stemmer)

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        aggregates: Dict[str, List[float]] = {}
        for pred, ref in zip(predictions, references):
            # Ensure both prediction and reference are non-empty strings
            pred_clean = pred.strip() if pred else "No prediction available."
            ref_clean = ref.strip() if ref else "No reference available."
            
            # Skip empty predictions/references to avoid ROUGE warnings
            if not pred_clean or not ref_clean:
                logger.warning("Empty prediction or reference detected, skipping...")
                continue
                
            scores = self.scorer.score(ref_clean, pred_clean)
            for rouge_type in self.rouge_types:
                for attr in ("precision", "recall", "fmeasure"):
                    key = f"{rouge_type}_{attr}"
                    aggregates.setdefault(key, []).append(getattr(scores[rouge_type], attr))

        return {key: float(np.mean(values)) for key, values in aggregates.items()}


class BERTScoreEvaluator:
    """Wrapper around `bert-score`."""

    def __init__(self, model_type: str, lang: str) -> None:
        if not BERTSCORE_AVAILABLE:
            raise ImportError("bert-score is not installed")
        self.model_type = model_type
        self.lang = lang

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        precision, recall, f1 = bert_score(predictions, references, model_type=self.model_type, lang=self.lang)
        return {
            'precision': float(precision.mean()),
            'recall': float(recall.mean()),
            'f1': float(f1.mean()),
        }


class TemporalCoherenceEvaluator:
    """Light-weight temporal coherence proxy based on event ordering consistency."""

    def __init__(self, data_loader: DataLoader) -> None:
        self.data_loader = data_loader

    def evaluate(self, predictions: List[str], events: List[Event]) -> float:
        if not predictions or not events:
            return 0.0
        # Extremely simple proxy: penalise empty outputs and reward coverage length
        scores = []
        for pred in predictions:
            tokens = pred.split()
            scores.append(min(len(tokens) / 200.0, 1.0))
        return float(np.mean(scores))


class TAEGEvaluator:
    """Main orchestrator for evaluating TAEG and baseline models."""

    def __init__(self, config: EvaluationConfig, data_loader: DataLoader) -> None:
        self.config = config
        self.data_loader = data_loader

        self._setup_directories()

        self.rouge_evaluator: Optional[ROUGEEvaluator] = None
        if self.config.compute_rouge and ROUGE_AVAILABLE:
            self.rouge_evaluator = ROUGEEvaluator(self.config.rouge_types, self.config.rouge_use_stemmer)

        self.bertscore_evaluator: Optional[BERTScoreEvaluator] = None
        if self.config.compute_bertscore and BERTSCORE_AVAILABLE:
            self.bertscore_evaluator = BERTScoreEvaluator(self.config.bertscore_model, self.config.bertscore_lang)

        self.temporal_evaluator: Optional[TemporalCoherenceEvaluator] = None
        if self.config.compute_temporal_coherence:
            self.temporal_evaluator = TemporalCoherenceEvaluator(data_loader)

        self._graph_builder: Optional[TAEGGraphBuilder] = None
        self._graph_cache: Optional[GeometricData] = None

    def _setup_directories(self) -> None:
        for directory in [self.config.output_dir, self.config.plot_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _get_full_graph(self) -> GeometricData:
        if not self.data_loader.events:
            self.data_loader.load_all_data()

        if self._graph_cache is None:
            self._graph_builder = TAEGGraphBuilder(self.data_loader)
            self._graph_cache = self._graph_builder.build_graph(include_temporal_edges=True)
        return self._graph_cache

    def evaluate_model(
        self,
        model: BaseModel,
        model_name: str,
        test_data: Any,
        reference_summaries: List[str],
    ) -> ModelResults:
        logger.info("Evaluating model %s", model_name)
        import time

        start_time = time.time()
        predictions = self._generate_predictions(model, test_data)
        inference_time = time.time() - start_time

        results = ModelResults(model_name=model_name, predictions=predictions, inference_time=inference_time)

        if self.rouge_evaluator and reference_summaries:
            try:
                results.rouge_scores = self.rouge_evaluator.evaluate(predictions, reference_summaries)
            except Exception as exc:  # pragma: no cover - metric errors should not break pipeline
                logger.error("ROUGE computation failed: %s", exc)

        if self.bertscore_evaluator and reference_summaries:
            try:
                results.bertscore_scores = self.bertscore_evaluator.evaluate(predictions, reference_summaries)
            except Exception as exc:  # pragma: no cover
                logger.error("BERTScore computation failed: %s", exc)

        if self.temporal_evaluator and hasattr(test_data, 'events'):
            try:
                results.temporal_coherence_score = self.temporal_evaluator.evaluate(predictions, test_data.events)
            except Exception as exc:  # pragma: no cover
                logger.error("Temporal coherence computation failed: %s", exc)

        return results

    def _generate_predictions(self, model: BaseModel, test_data: Any) -> List[str]:
        predictions: List[str] = []

        if isinstance(model, TAEGModel):
            graph_data = self._get_full_graph()
            for idx, _ in enumerate(test_data.events):
                try:
                    prediction = model.generate_summary((graph_data, idx))
                    # Ensure prediction is never empty
                    if not prediction or not prediction.strip():
                        prediction = "No summary generated."
                    predictions.append(prediction)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to generate summary for event %d: %s", idx, exc)
                    predictions.append("Error during summary generation.")
        else:
            for event in test_data.events:
                text = self.data_loader.get_concatenated_text_for_event(event)
                try:
                    if hasattr(model, 'generate_summary'):
                        prediction = model.generate_summary(text)
                        # Ensure prediction is never empty
                        if not prediction or not prediction.strip():
                            prediction = "No summary generated."
                    else:
                        prediction = "Model does not support summarization."
                except Exception as exc:  # pragma: no cover
                    logger.warning("Baseline generation error: %s", exc)
                    prediction = "Error during summary generation."
                predictions.append(prediction)
        return predictions

    def compare_models(self, results: List[ModelResults]) -> Dict[str, Any]:
        comparison: Dict[str, Any] = {'model_names': [r.model_name for r in results], 'metrics_comparison': {}}

        if results and results[0].rouge_scores:
            for metric in results[0].rouge_scores.keys():
                metric_values = {r.model_name: (r.rouge_scores or {}).get(metric) for r in results}
                best = max(
                    (name for name, value in metric_values.items() if value is not None),
                    key=lambda n: metric_values[n],
                    default=None,
                )
                comparison['metrics_comparison'][metric] = {
                    'values': metric_values,
                    'best_model': best,
                }
        return comparison

    def create_evaluation_plots(self, results: List[ModelResults], comparison: Dict[str, Any]) -> None:
        if not self.config.create_plots or not PLOT_LIBS_AVAILABLE or not results:
            return

        rougel = [
            (r.model_name, (r.rouge_scores or {}).get('rougeL_fmeasure', 0.0))
            for r in results
        ]
        df = pd.DataFrame(rougel, columns=['model', 'rougeL_fmeasure'])

        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x='model', y='rougeL_fmeasure', palette='Blues_d')
        plt.ylabel('ROUGE-L F1')
        plt.xlabel('Model')
        plt.ylim(0, 1)
        plt.tight_layout()

        plot_path = Path(self.config.plot_dir) / 'rougeL_f1.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info("Saved evaluation plot to %s", plot_path)

    def save_results(self, results: List[ModelResults], comparison: Dict[str, Any]) -> None:
        if not self.config.save_results:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            payload = {
                'model_name': result.model_name,
                'rouge_scores': result.rouge_scores,
                'bertscore_scores': result.bertscore_scores,
                'temporal_coherence_score': result.temporal_coherence_score,
                'inference_time': result.inference_time,
            }
            with open(output_dir / f"{result.model_name}_results.json", 'w') as handle:
                json.dump(payload, handle, indent=2)

            if self.config.save_predictions:
                with open(output_dir / f"{result.model_name}_predictions.txt", 'w') as handle:
                    for idx, pred in enumerate(result.predictions, start=1):
                        handle.write(f"Event {idx}:\n{pred}\n\n")

        with open(output_dir / 'model_comparison.json', 'w') as handle:
            json.dump(comparison, handle, indent=2)
        logger.info("Saved evaluation artefacts to %s", output_dir)


def main() -> None:  # pragma: no cover - illustrative usage
    logging.basicConfig(level=logging.INFO)
    data_loader = DataLoader("data")
    data_loader.load_all_data()

    config = EvaluationConfig()
    evaluator = TAEGEvaluator(config, data_loader)

    class DummyTestData:
        def __init__(self, events: List[Event]) -> None:
            self.events = events

    test_data = DummyTestData(data_loader.events)
    dummy_references = [f"Reference summary for event {event.event_id}" for event in data_loader.events]

    # Demonstration with LexRank (extractive baseline)
    from .models import ModelFactory

    lexrank = ModelFactory.create_model('lexrank', num_sentences=3)
    results = evaluator.evaluate_model(lexrank, 'LexRank', test_data, dummy_references)
    comparison = evaluator.compare_models([results])
    evaluator.save_results([results], comparison)
    if config.create_plots:
        evaluator.create_evaluation_plots([results], comparison)


if __name__ == "__main__":  # pragma: no cover
    main()
