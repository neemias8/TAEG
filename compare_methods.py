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
        ("lexrank", 750, "LEXRANK (750 sentences total)"),
        ("lexrank-ta", 1, "LEXRANK-TA (optimized temporal anchoring)")
    ]

    results = {}

    for method, length, description in methods:
        print(f"\nTesting: {description}")
        print(f"Configuration: method={method}, length={length}")
        print("-"*60)

        # Create a fresh pipeline instance for each test
        pipeline = TAEGPipeline()
        result = pipeline.run_pipeline(summary_length=length, summarization_method=method)

        if "error" in result:
            print(f"ERROR with {method}: {result['error']}")
            continue

        # Extract key metrics
        rouge_f1 = result["evaluation"]["rouge"]["rouge1"]["f1"]
        rouge2_f1 = result["evaluation"]["rouge"]["rouge2"]["f1"]
        rougeL_f1 = result["evaluation"]["rouge"]["rougeL"]["f1"]
        bert_f1 = result["evaluation"]["bertscore"]["f1"]
        meteor = result["evaluation"]["meteor"]
        kendall_tau = result["evaluation"]["kendall_tau"]

        # Store results with unique key including parameters
        result_key = f"{method}_{length}"
        results[result_key] = {
            "method": method,
            "length": length,
            "description": description,
            "summary_length_chars": len(result["consolidated_summary"]),
            "summary_sentences": result["consolidated_summary"].count('.'),
            "rouge1_f1": rouge_f1,
            "rouge2_f1": rouge2_f1,
            "rougeL_f1": rougeL_f1,
            "bertscore_f1": bert_f1,
            "meteor": meteor,
            "kendall_tau": kendall_tau,
            "summary_preview": result["consolidated_summary"][:150] + "...",
            "summary_hash": hash(result["consolidated_summary"])
        }

        print(f"Length: {len(result['consolidated_summary'])} characters")
        print(f"Sentences: ~{result['consolidated_summary'].count('.')} sentences")
        print(f"ROUGE-1 F1: {rouge_f1:.3f}")
        print(f"ROUGE-2 F1: {rouge2_f1:.3f}")
        print(f"ROUGE-L F1: {rougeL_f1:.3f}")
        print(f"BERTScore F1: {bert_f1:.3f}")
        print(f"METEOR: {meteor:.3f}")
        print(f"Kendall's Tau: {kendall_tau:.3f}")
        print(f"Summary Hash: {results[result_key]['summary_hash']}")
        print(f"Preview: {results[result_key]['summary_preview']}")

    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    # Find lexrank and lexrank-ta results
    lexrank_key = next((k for k in results.keys() if k.startswith("lexrank_")), None)
    ta_key = next((k for k in results.keys() if k.startswith("lexrank-ta_")), None)

    if lexrank_key and ta_key and len(results) == 2:
        lexrank_result = results[lexrank_key]
        ta_result = results[ta_key]

        print(f"\nConfigurations tested:")
        print(f"  LEXRANK:     {lexrank_result['method']} (length={lexrank_result['length']})")
        print(f"  LEXRANK-TA:  {ta_result['method']} (length={ta_result['length']})")

        print(f"\nSummary verification:")
        print(f"  LEXRANK Hash:     {lexrank_result['summary_hash']}")
        print(f"  LEXRANK-TA Hash:  {ta_result['summary_hash']}")
        
        if lexrank_result['summary_hash'] == ta_result['summary_hash']:
            print("  WARNING: Summaries are identical! Possible caching issue.")
        else:
            print("  OK: Summaries are different.")

        print(f"\nROUGE-1 F1:")
        print(f"  LEXRANK:     {lexrank_result['rouge1_f1']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['rouge1_f1']:.3f}")
        diff_rouge1 = ta_result['rouge1_f1'] - lexrank_result['rouge1_f1']
        pct_rouge1 = (diff_rouge1 / lexrank_result['rouge1_f1']) * 100 if lexrank_result['rouge1_f1'] != 0 else 0
        print(f"  Difference:  {diff_rouge1:+.3f} ({pct_rouge1:+.1f}%)")

        print(f"\nROUGE-2 F1:")
        print(f"  LEXRANK:     {lexrank_result['rouge2_f1']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['rouge2_f1']:.3f}")
        diff_rouge2 = ta_result['rouge2_f1'] - lexrank_result['rouge2_f1']
        pct_rouge2 = (diff_rouge2 / lexrank_result['rouge2_f1']) * 100 if lexrank_result['rouge2_f1'] != 0 else 0
        print(f"  Difference:  {diff_rouge2:+.3f} ({pct_rouge2:+.1f}%)")

        print(f"\nROUGE-L F1:")
        print(f"  LEXRANK:     {lexrank_result['rougeL_f1']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['rougeL_f1']:.3f}")
        diff_rougeL = ta_result['rougeL_f1'] - lexrank_result['rougeL_f1']
        pct_rougeL = (diff_rougeL / lexrank_result['rougeL_f1']) * 100 if lexrank_result['rougeL_f1'] != 0 else 0
        print(f"  Difference:  {diff_rougeL:+.3f} ({pct_rougeL:+.1f}%)")

        print(f"\nBERTScore F1:")
        print(f"  LEXRANK:     {lexrank_result['bertscore_f1']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['bertscore_f1']:.3f}")
        diff_bert = ta_result['bertscore_f1'] - lexrank_result['bertscore_f1']
        pct_bert = (diff_bert / lexrank_result['bertscore_f1']) * 100 if lexrank_result['bertscore_f1'] != 0 else 0
        print(f"  Difference:  {diff_bert:+.3f} ({pct_bert:+.1f}%)")

        print(f"\nMETEOR:")
        print(f"  LEXRANK:     {lexrank_result['meteor']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['meteor']:.3f}")
        diff_meteor = ta_result['meteor'] - lexrank_result['meteor']
        pct_meteor = (diff_meteor / lexrank_result['meteor']) * 100 if lexrank_result['meteor'] != 0 else 0
        print(f"  Difference:  {diff_meteor:+.3f} ({pct_meteor:+.1f}%)")

        print(f"\nKendall's Tau (temporal order):")
        print(f"  LEXRANK:     {lexrank_result['kendall_tau']:.3f}")
        print(f"  LEXRANK-TA:  {ta_result['kendall_tau']:.3f}")
        diff_kendall = ta_result['kendall_tau'] - lexrank_result['kendall_tau']
        pct_kendall = (diff_kendall / abs(lexrank_result['kendall_tau'])) * 100 if lexrank_result['kendall_tau'] != 0 else 0
        print(f"  Difference:  {diff_kendall:+.3f} ({pct_kendall:+.1f}%)")

        print(f"\nSummary length:")
        print(f"  LEXRANK:     {lexrank_result['summary_length_chars']} chars, {lexrank_result['summary_sentences']} sentences")
        print(f"  LEXRANK-TA:  {ta_result['summary_length_chars']} chars, {ta_result['summary_sentences']} sentences")

        # Analysis
        print(f"\nANALYSIS:")
        if abs(ta_result['kendall_tau']) > abs(lexrank_result['kendall_tau']):
            print("  LEXRANK-TA has better temporal preservation!")
        else:
            print("  LEXRANK still has better temporal order")

        if ta_result['rouge1_f1'] > lexrank_result['rouge1_f1']:
            print("  LEXRANK-TA has better semantic quality (ROUGE-1)!")
        else:
            print("  LEXRANK has better semantic quality (ROUGE-1)")

        if ta_result['rouge2_f1'] > lexrank_result['rouge2_f1']:
            print("  LEXRANK-TA has better semantic quality (ROUGE-2)!")
        else:
            print("  LEXRANK has better semantic quality (ROUGE-2)")

        if ta_result['rougeL_f1'] > lexrank_result['rougeL_f1']:
            print("  LEXRANK-TA has better semantic quality (ROUGE-L)!")
        else:
            print("  LEXRANK has better semantic quality (ROUGE-L)")

        if ta_result['bertscore_f1'] > lexrank_result['bertscore_f1']:
            print("  LEXRANK-TA has better semantic quality (BERT)!")
        else:
            print("  LEXRANK has better semantic quality (BERT)")

    else:
        print("ERROR: Cannot compare - insufficient results")
        print(f"Available results: {list(results.keys())}")

if __name__ == "__main__":
    compare_methods()