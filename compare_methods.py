#!/usr/bin/env python3
"""
Comparison script for LEXRANK vs LEXRANK-TA methods
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import TAEGPipeline

def compare_methods():
    """Compare LEXRANK vs LEXRANK-TA methods."""
    print("TAEG - Method Comparison: LEXRANK vs LEXRANK-TA")
    print("="*80)

    methods = [
        ("lexrank", 338, "LEXRANK (338 sentences total)"),
        ("lexrank-ta", 2, "LEXRANK-TA (2 sentences per event)"),
        ("lexrank-ta-best", 1, "LEXRANK-TA-BEST (best gospel per event)")
    ]

    results = {}

    for method, length, description in methods:
        print(f"\n🧪 Testing: {description}")
        print("-"*60)

        pipeline = TAEGPipeline()
        result = pipeline.run_pipeline(summary_length=length, summarization_method=method)

        if "error" in result:
            print(f"❌ Error with {method}: {result['error']}")
            continue

        # Extract key metrics
        rouge_f1 = result["evaluation"]["rouge"]["rouge1"]["f1"]
        bert_f1 = result["evaluation"]["bertscore"]["f1"]
        meteor = result["evaluation"]["meteor"]
        kendall_tau = result["evaluation"]["kendall_tau"]

        results[method] = {
            "description": description,
            "summary_length_chars": len(result["consolidated_summary"]),
            "rouge1_f1": rouge_f1,
            "bertscore_f1": bert_f1,
            "meteor": meteor,
            "kendall_tau": kendall_tau,
            "summary_preview": result["consolidated_summary"][:150] + "..."
        }

        print(f"📏 Tamanho: {len(result['consolidated_summary'])} caracteres")
        print(f"🎯 ROUGE-1 F1: {rouge_f1:.3f}")
        print(f"🤖 BERTScore F1: {bert_f1:.3f}")
        print(f"📊 METEOR: {meteor:.3f}")
        print(f"⏰ Kendall's Tau: {kendall_tau:.3f}")
        print(f"📝 Preview: {results[method]['summary_preview']}")

    # Summary comparison
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)

    if len(results) == 2:
        lexrank_result = results["lexrank"]
        ta_result = results["lexrank-ta"]

        print("\n🎯 ROUGE-1 F1:")
        print(f"  LEXRANK:     {lexrank_result['rouge1_f1']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['rouge1_f1']:.3f}")
        print(f"  Diferença:   {ta_result['rouge1_f1'] - lexrank_result['rouge1_f1']:+.3f}")

        print("\n🤖 BERTScore F1:")
        print(f"  LEXRANK:     {lexrank_result['bertscore_f1']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['bertscore_f1']:.3f}")
        print(f"  Diferença:   {ta_result['bertscore_f1'] - lexrank_result['bertscore_f1']:+.3f}")

        print("\n📊 METEOR:")
        print(f"  LEXRANK:     {lexrank_result['meteor']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['meteor']:.3f}")
        print(f"  Diferença:   {ta_result['meteor'] - lexrank_result['meteor']:+.3f}")

        print("\n⏰ Kendall's Tau (ordem temporal):")
        print(f"  LEXRANK:     {lexrank_result['kendall_tau']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['kendall_tau']:.3f}")
        print(f"  Diferença:   {ta_result['kendall_tau'] - lexrank_result['kendall_tau']:+.3f}")

        print("\n📏 Tamanho:")
        print(f"  LEXRANK:     {lexrank_result['summary_length_chars']} chars")
        print(f"  LEXRANK-TA:  {ta_result['summary_length_chars']} chars")

        # Analysis
        print("\n🔍 ANÁLISE:")
        if abs(ta_result['kendall_tau']) < abs(lexrank_result['kendall_tau']):
            print("  ✅ LEXRANK-TA tem melhor preservação temporal!")
        else:
            print("  ❌ LEXRANK ainda tem melhor ordem temporal")

        if ta_result['rouge1_f1'] > lexrank_result['rouge1_f1']:
            print("  ✅ LEXRANK-TA tem melhor qualidade semântica!")
        else:
            print("  ❌ LEXRANK tem melhor qualidade semântica")

if __name__ == "__main__":
    compare_methods()