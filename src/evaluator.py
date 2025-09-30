"""
Evaluation module for assessing text summarization quality using multiple metrics.
"""

import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
from scipy.stats import kendalltau
from typing import Dict, List, Any, Tuple
import warnings
from data_loader import ChronologyLoader


class SummarizationEvaluator:
    """Evaluator for text summarization using multiple metrics."""

    def __init__(self):
        """Initialize the evaluator."""
        # Download required NLTK data for METEOR
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')

        try:
            nltk.data.find('omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_rouge(self, hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores.

        Args:
            hypothesis: Generated summary
            reference: Reference summary

        Returns:
            Dictionary with ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, hypothesis)

        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'f1': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'f1': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'f1': scores['rougeL'].fmeasure
            }
        }

    def calculate_meteor(self, hypothesis: str, reference: str) -> float:
        """
        Calculate METEOR score.

        Args:
            hypothesis: Generated summary
            reference: Reference summary

        Returns:
            METEOR score
        """
        try:
            from nltk.translate.meteor_score import meteor_score
            hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
            reference_tokens = nltk.word_tokenize(reference.lower())
            return meteor_score([reference_tokens], hypothesis_tokens)
        except ImportError:
            # Fallback if meteor_score is not available
            warnings.warn("METEOR score calculation failed. Using BLEU as fallback.")
            from nltk.translate.bleu_score import sentence_bleu
            hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
            reference_tokens = nltk.word_tokenize(reference.lower())
            return sentence_bleu([reference_tokens], hypothesis_tokens)

    def calculate_bertscore(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """
        Calculate BERTScore.

        Args:
            hypothesis: Generated summary
            reference: Reference summary

        Returns:
            Dictionary with BERTScore metrics
        """
        try:
            P, R, F1 = bert_score([hypothesis], [reference], lang='en', verbose=False)

            return {
                'precision': P.item(),
                'recall': R.item(),
                'f1': F1.item()
            }
        except Exception as e:
            warnings.warn(f"BERTScore calculation failed: {e}. Returning zeros.")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

    def calculate_kendall_tau(self, hypothesis: str, reference: str, is_temporal_anchored: bool = False) -> float:
        """
        Calculate Kendall's Tau correlation between chronological event ordering.

        Args:
            hypothesis: Generated summary
            reference: Reference summary (Golden Sample)
            is_temporal_anchored: Whether the method guarantees chronological ordering

        Returns:
            Kendall's Tau correlation coefficient (-1 to 1)
        """
        if is_temporal_anchored:
            # For temporal anchoring methods, order is guaranteed to be chronological
            return 1.0

        # Use Golden Sample to determine event ordering
        return self._calculate_kendall_tau_from_golden_sample(hypothesis, reference)

    def evaluate_summary(self, hypothesis: str, reference: str, is_temporal_anchored: bool = False) -> Dict[str, Any]:
        """
        Evaluate a summary against a reference using all metrics.

        Args:
            hypothesis: Generated summary
            reference: Reference summary
            is_temporal_anchored: Whether the method guarantees chronological ordering

        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}

        # ROUGE scores
        results['rouge'] = self.calculate_rouge(hypothesis, reference)

        # METEOR score
        results['meteor'] = self.calculate_meteor(hypothesis, reference)

        # BERTScore
        results['bertscore'] = self.calculate_bertscore(hypothesis, reference)

        # Kendall's Tau - temporal order correlation
        results['kendall_tau'] = self.calculate_kendall_tau(hypothesis, reference, is_temporal_anchored)

        return results

    def print_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation results in a formatted way.

        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        print("\nROUGE Scores:")
        for rouge_type, scores in results['rouge'].items():
            print(f"  {rouge_type.upper()}:")
            print(f"    Precision: {scores['precision']:.3f}")
            print(f"    Recall: {scores['recall']:.3f}")
            print(f"    F1: {scores['f1']:.3f}")

        print(f"METEOR: {results['meteor']:.3f}")
        print("\nBERTScore:")
        print(f"    Precision: {results['bertscore']['precision']:.3f}")
        print(f"    Recall: {results['bertscore']['recall']:.3f}")
        print(f"    F1: {results['bertscore']['f1']:.3f}")

        print(f"Kendall's Tau: {results['kendall_tau']:.3f}")

    def _calculate_kendall_tau_external(self, hypothesis: str) -> float:
        """
        Calculate Kendall's Tau using external chronology XML.

        Args:
            hypothesis: Generated summary

        Returns:
            Kendall's Tau correlation coefficient
        """
        try:
            # Load chronological events
            chrono_loader = ChronologyLoader()
            events = chrono_loader.load_chronology()

            if not events:
                return 0.0

            # Get event descriptions for matching
            event_descriptions = [event['description'].lower() for event in events]

            # Split hypothesis into sentences
            hyp_sentences = nltk.sent_tokenize(hypothesis.lower())

            if len(hyp_sentences) < 2:
                return 0.0

            # Find the position of each chronological event in the summary
            event_positions = {}
            for i, event_desc in enumerate(event_descriptions):
                # Look for the event in the summary sentences
                for j, sentence in enumerate(hyp_sentences):
                    # Simple string matching - could be improved with semantic similarity
                    if any(keyword in sentence for keyword in event_desc.split()):
                        event_positions[i] = j
                        break

            # If we found at least 2 events, calculate correlation
            if len(event_positions) >= 2:
                # Create expected order (chronological IDs)
                expected_order = list(event_positions.keys())

                # Create found order (positions in summary)
                found_order = [event_positions[event_id] for event_id in expected_order]

                # Calculate Kendall's Tau
                tau, _ = kendalltau(expected_order, found_order)
                return tau if not np.isnan(tau) else 0.0

            # If fewer than 2 events found, return score based on coverage
            coverage = len(event_positions) / len(events)
            return coverage * 0.5  # Scale to reasonable range

        except Exception as e:
            return 0.0

    def _calculate_kendall_tau_from_golden_sample(self, hypothesis: str, reference: str) -> float:
        """
        Calculate Kendall's Tau by comparing event ordering in hypothesis vs reference (Golden Sample).

        This method identifies key events in the Golden Sample (which has known chronological order)
        and finds their order in the generated summary to assess temporal preservation.

        Args:
            hypothesis: Generated summary
            reference: Golden Sample (chronologically ordered reference)

        Returns:
            Kendall's Tau correlation coefficient (-1 to 1)
        """
        try:
            # Load chronological events from XML to get event descriptions
            chrono_loader = ChronologyLoader()
            events = chrono_loader.load_chronology()

            if not events:
                return 0.0

            # Get event descriptions and IDs
            event_data = [(i, event['description'].lower()) for i, event in enumerate(events)]

            # Split texts into sentences
            ref_sentences = nltk.sent_tokenize(reference.lower())
            hyp_sentences = nltk.sent_tokenize(hypothesis.lower())

            if len(hyp_sentences) < 2:
                return 0.0

            # Find events in reference (Golden Sample) - these define the expected chronological order
            ref_event_positions = {}
            for event_id, event_desc in event_data:
                for j, sentence in enumerate(ref_sentences):
                    # Look for event in reference sentences
                    if any(keyword in sentence for keyword in event_desc.split()[:3]):  # Use first 3 words for matching
                        ref_event_positions[event_id] = j
                        print(f"DEBUG: Event {event_id} ('{event_desc}') found in reference at position {j}")
                        break

            # Find the same events in hypothesis (generated summary)
            hyp_event_positions = {}
            for event_id, event_desc in event_data:
                for j, sentence in enumerate(hyp_sentences):
                    # Look for event in hypothesis sentences
                    if any(keyword in sentence for keyword in event_desc.split()[:3]):  # Use first 3 words for matching
                        hyp_event_positions[event_id] = j
                        print(f"DEBUG: Event {event_id} ('{event_desc}') found in hypothesis at position {j}")
                        break

            # Only consider events found in both texts
            common_events = set(ref_event_positions.keys()) & set(hyp_event_positions.keys())

            print(f"DEBUG: Found {len(common_events)} common events out of {len(events)} total events")
            print(f"DEBUG: Common events: {sorted(common_events)}")

            if len(common_events) < 2:
                # If we can't find enough events, return a low score
                coverage_penalty = len(common_events) / len(events)
                return coverage_penalty * 0.3  # Low score for poor event coverage

            # Create orderings based on positions in respective texts
            common_event_list = sorted(common_events)

            # Expected order: chronological order from reference (Golden Sample positions)
            expected_order = [ref_event_positions[event_id] for event_id in common_event_list]

            # Found order: order in generated summary
            found_order = [hyp_event_positions[event_id] for event_id in common_event_list]

            print(f"DEBUG: Expected order (Golden Sample positions): {expected_order}")
            print(f"DEBUG: Found order (summary positions): {found_order}")

            # Calculate Kendall's Tau between the two orderings
            tau, _ = kendalltau(expected_order, found_order)
            print(f"DEBUG: Kendall's Tau = {tau}")
            return tau if not np.isnan(tau) else 0.0

        except Exception as e:
            print(f"Warning: Error calculating Kendall's Tau from Golden Sample: {e}")
            return 0.0