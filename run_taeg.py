#!/usr/bin/env python3
"""
Simple runner script for TAEG pipeline.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import TAEGPipeline

def main():
    """Run the TAEG pipeline with default settings."""
    import argparse

    parser = argparse.ArgumentParser(description="TAEG - Text Analysis of Evangelic Gospels")
    parser.add_argument("--method", choices=["lexrank", "lexrank-ta", "lexrank-ta-best"],
                       default="lexrank",
                       help="Summarization method: lexrank (original), lexrank-ta (temporal anchoring with summary), or lexrank-ta-best (temporal anchoring with best gospel)")
    parser.add_argument("--length", type=int, default=500, help="Number of sentences in summary (or sentences per event for lexrank-ta)")

    args = parser.parse_args()

    print("TAEG - Text Analysis of Evangelic Gospels")
    print(f"Consolidating gospel narratives using {args.method.upper()}")

    pipeline = TAEGPipeline()
    results = pipeline.run_pipeline(summary_length=args.length, summarization_method=args.method)

    if "error" not in results:
        pipeline.save_results(results)
        print("\n✅ Pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()