"""
Main module for the TAEG (Text Analysis of Evangelic Gospels) project.
Combines multiple gospels into a consolidated summary using LEXRANK and evaluates it.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import BiblicalDataLoader
from summarizer import LexRankSummarizer, LexRankTemporalAnchoring
from evaluator import SummarizationEvaluator


class TAEGPipeline:
    """Main pipeline for TAEG project."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the TAEG pipeline.

        Args:
            data_dir: Directory containing the data files
        """
        self.data_loader = BiblicalDataLoader(data_dir)
        self.summarizer_lexrank = LexRankSummarizer()
        self.summarizer_ta = LexRankTemporalAnchoring()
        self.evaluator = SummarizationEvaluator()

    def run_pipeline(self, summary_length: int = 500, summarization_method: str = "lexrank") -> Dict[str, Any]:
        """
        Run the complete TAEG pipeline using multi-document LEXRANK.

        Args:
            summary_length: Number of sentences in the final summary

        Returns:
            Dictionary with results and evaluation metrics
        """
        print("Starting TAEG Pipeline...")
        print("="*50)
        print("Using multi-document LEXRANK summarization")

        # Step 1: Load data
        print("\n1. Loading biblical texts...")
        try:
            gospel_texts = self.data_loader.load_all_gospels()
            golden_sample = self.data_loader.load_golden_sample()

            print(f"   Loaded texts from {len(gospel_texts)} gospels")
            print(f"   Golden sample length: {len(golden_sample)} characters")

            for gospel, text in gospel_texts.items():
                print(f"   {gospel}: {len(text)} characters")

        except Exception as e:
            print(f"Error loading data: {e}")
            return {"error": str(e)}

        # Step 2: Create consolidated summary
        print(f"\n2. Creating consolidated summary using {summarization_method.upper()}...")
        try:
            if summarization_method.lower() == "lexrank":
                texts_list = list(gospel_texts.values())
                consolidated_summary = self.summarizer_lexrank.summarize_texts(texts_list, summary_length)
            elif summarization_method.lower() == "lexrank-ta":
                # For LEXRANK-TA, summary_length represents sentences per event
                consolidated_summary = self.summarizer_ta.summarize_with_temporal_anchoring(summary_length)
            elif summarization_method.lower() == "lexrank-ta-best":
                # For LEXRANK-TA-BEST, use the best gospel for each event
                consolidated_summary = self.summarizer_ta.summarize_with_temporal_anchoring(summary_length, use_best_gospel=True)
            else:
                raise ValueError(f"Unknown summarization method: {summarization_method}")

            print(f"   Generated summary with {len(consolidated_summary)} characters")
            print(f"   Summary preview: {consolidated_summary[:200]}...")

        except Exception as e:
            print(f"Error creating summary: {e}")
            return {"error": str(e)}

        # Step 3: Evaluate summary
        print("\n3. Evaluating summary against Golden Sample...")
        try:
            # Determine if this is a temporal anchoring method
            is_temporal_anchored = summarization_method.lower() in ["lexrank-ta", "lexrank-ta-best"]
            evaluation_results = self.evaluator.evaluate_summary(consolidated_summary, golden_sample, is_temporal_anchored)

            print("   Evaluation completed successfully")

        except Exception as e:
            print(f"Error evaluating summary: {e}")
            return {"error": str(e)}

        # Step 4: Print results
        print("\n4. Results:")
        self.evaluator.print_evaluation_results(evaluation_results)

        # Prepare final results
        results = {
            "gospel_texts": gospel_texts,
            "golden_sample": golden_sample,
            "consolidated_summary": consolidated_summary,
            "evaluation": evaluation_results,
            "summary_length": summary_length,
            "summarization_method": summarization_method
        }

        return results

    def save_results(self, results: Dict[str, Any], output_dir: str = "outputs") -> None:
        """
        Save results to files.

        Args:
            results: Results dictionary from run_pipeline
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Get method name for file naming
        method = results.get("summarization_method", "lexrank").upper()

        # Save consolidated summary with method-specific name
        summary_file = output_path / f"taeg_summary_{method.lower()}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(results["consolidated_summary"])

        # Save evaluation results with method-specific name
        eval_file = output_path / f"evaluation_results_{method.lower()}.json"
        import json
        with open(eval_file, 'w', encoding='utf-8') as f:
            # Convert numpy types to Python types for JSON serialization
            eval_results = results["evaluation"].copy()
            # Add method info to evaluation results
            eval_results["summarization_method"] = method
            json.dump(eval_results, f, indent=2)

        print(f"\nResults saved to {output_path}/")
        print(f"  📄 Summary: {summary_file.name}")
        print(f"  📊 Evaluation: {eval_file.name}")


def main():
    """Main function to run the TAEG pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="TAEG - Text Analysis of Evangelic Gospels")
    parser.add_argument("--data-dir", default="data", help="Directory containing data files")
    parser.add_argument("--summary-length", type=int, default=500, help="Number of sentences in summary (or sentences per event for lexrank-ta)")
    parser.add_argument("--method", choices=["lexrank", "lexrank-ta"],
                       default="lexrank",
                       help="Summarization method: lexrank (original) or lexrank-ta (temporal anchoring)")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save results")

    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = TAEGPipeline(args.data_dir)
    results = pipeline.run_pipeline(args.summary_length, args.method)

    if "error" not in results:
        pipeline.save_results(results, args.output_dir)
        print("\nPipeline completed successfully!")
    else:
        print(f"\nPipeline failed: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()