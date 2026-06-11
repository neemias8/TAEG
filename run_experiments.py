#!/usr/bin/env python3
"""
Unified experiment runner for the JBCS revision (Task 5 of
docs/JBCS_REVISION_SPEC.md).

One command reproduces every number in the revised paper:

    python run_experiments.py --all

Runs the timeline-agnostic LexRank baseline (750 sentences, as in the
paper), every timeline-aware selection strategy (longest, random x N seeds,
priority, centroid, taeg), both TAEG ablations, and the timeline
degradation experiment. Every method is evaluated by the SAME evaluator
instance against the SAME Golden Sample (bit-for-bit comparable).

Outputs (in --output-dir, default outputs/):
    results_all_methods.json   all metrics + config + git commit hash
    results_table.md           consolidated table (markdown)
    results_table.tex          LaTeX rows for the paper's tables
    results_degradation.json   Task 4 report
    ablation_divergence.json   Task 3 report
    selection_report_<m>.json  per-event selection report per method
    summary_<m>.txt            consolidated narrative per deterministic method
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

# Windows cp1252 consoles crash on the emoji prints used across the
# codebase; force UTF-8 so the runner works regardless of console encoding.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from data_loader import BiblicalDataLoader, ChronologyLoader  # noqa: E402
from degradation import DEGRADATION_LEVELS, DEGRADATION_SEEDS, consolidate_degraded  # noqa: E402
from evaluator import SummarizationEvaluator  # noqa: E402
from improved_graph_builder import ImprovedTemporalGraphBuilder  # noqa: E402
from selection_strategies import get_strategy, selection_divergence  # noqa: E402
from summarizer import LexRankSummarizer, LexRankTemporalAnchoring  # noqa: E402
from taeg_centrality import ablation_flags, compute_taeg_centrality  # noqa: E402

LEXRANK_LENGTH = 750  # sentences, as in the published paper
RANDOM_SEEDS_DEFAULT = 30

DETERMINISTIC_STRATEGIES = ['longest', 'priority', 'centroid',
                            'taeg', 'taeg-no-before', 'taeg-no-same-event']
ALL_METHODS = ['lexrank'] + DETERMINISTIC_STRATEGIES + ['random']

METRIC_KEYS = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1',
               'bertscore_f1', 'meteor', 'kendall_tau']
METRIC_LABELS = {
    'rouge1_f1': 'ROUGE-1 F1', 'rouge2_f1': 'ROUGE-2 F1',
    'rougeL_f1': 'ROUGE-L F1', 'bertscore_f1': 'BERTScore F1',
    'meteor': 'METEOR', 'kendall_tau': "Kendall's Tau",
}


def flatten_metrics(ev):
    return {
        'rouge1_f1': ev['rouge']['rouge1']['f1'],
        'rouge2_f1': ev['rouge']['rouge2']['f1'],
        'rougeL_f1': ev['rouge']['rougeL']['f1'],
        'bertscore_f1': ev['bertscore']['f1'],
        'meteor': ev['meteor'],
        'kendall_tau': ev['kendall_tau'],
    }


def mean_std(per_seed_metrics):
    """Aggregate a list of flattened-metric dicts into mean/std (sample std)."""
    mean, std = {}, {}
    for key in METRIC_KEYS + ['length_chars']:
        values = [m[key] for m in per_seed_metrics]
        mean[key] = statistics.mean(values)
        std[key] = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def git_commit_hash():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


class ExperimentRunner:
    def __init__(self, output_dir: Path, random_seeds: int):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.random_seeds = random_seeds

        print("Loading data and building the TAEG (once, shared by all methods)...")
        self.data_loader = BiblicalDataLoader()
        self.gospel_texts = self.data_loader.load_all_gospels()
        self.golden = self.data_loader.load_golden_sample()
        self.events = ChronologyLoader().load_chronology()
        self.graph = ImprovedTemporalGraphBuilder().build_improved_temporal_graph(verbose=False)
        self.consolidator = LexRankTemporalAnchoring()
        self.evaluator = SummarizationEvaluator(verbose=False)
        self._centrality_cache = {}

        # Run artifacts kept for downstream analyses (Task 5b).
        self.summaries = {}           # method -> consolidated text
        self.records = {}             # method -> selection records
        self.random_runs = []         # [{seed, summary, records, metrics}]
        self.results = {'methods': {}}

    # ---------------- core helpers ----------------

    def _evaluate(self, summary: str) -> dict:
        ev = self.evaluator.evaluate_summary(summary, self.golden)
        flat = flatten_metrics(ev)
        flat['length_chars'] = len(summary)
        return {'full': ev, 'flat': flat}

    def _centrality_for(self, key: str):
        if key not in self._centrality_cache:
            self._centrality_cache[key] = compute_taeg_centrality(
                self.graph, **ablation_flags(key))
        return self._centrality_cache[key]

    def _strategy_for(self, key: str, seed=None):
        if key.startswith('taeg'):
            centrality, _ = self._centrality_for(key)
            return get_strategy(key, centrality=centrality)
        return get_strategy(key, seed=seed)

    def _write_json(self, name: str, payload: dict):
        path = self.output_dir / name
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {path}")

    # ---------------- method runners ----------------

    def run_lexrank(self):
        print(f"\n=== lexrank (timeline-agnostic baseline, {LEXRANK_LENGTH} sentences) ===")
        t0 = time.time()
        summary = LexRankSummarizer().summarize_texts(
            list(self.gospel_texts.values()), LEXRANK_LENGTH)
        evaluated = self._evaluate(summary)
        self.summaries['lexrank'] = summary
        self.results['methods']['lexrank'] = {
            'kind': 'timeline-agnostic',
            'config': {'summary_length_sentences': LEXRANK_LENGTH},
            'metrics': evaluated['flat'],
            'evaluation': evaluated['full'],
        }
        (self.output_dir / 'summary_lexrank.txt').write_text(summary, encoding='utf-8')
        print(f"  done in {time.time()-t0:.1f}s: {evaluated['flat']}")

    def run_deterministic_strategy(self, key: str):
        print(f"\n=== {key} (timeline-aware) ===")
        t0 = time.time()
        strategy = self._strategy_for(key)
        summary, records = self.consolidator.consolidate_with_strategy(
            strategy, graph=self.graph, events=self.events, verbose=False)
        evaluated = self._evaluate(summary)
        self.summaries[key] = summary
        self.records[key] = records
        entry = {
            'kind': 'timeline-aware',
            'metrics': evaluated['flat'],
            'evaluation': evaluated['full'],
        }
        if key.startswith('taeg'):
            _, info = self._centrality_for(key)
            entry['centrality_info'] = info
            n_not_longest = sum(
                1 for r in records
                if not r['fallback'] and r['n_candidates'] >= 2
                and next(c['text_length'] for c in r['candidates']
                         if c['node_id'] == r['chosen_node'])
                < max(c['text_length'] for c in r['candidates'])
            )
            entry['events_choosing_non_longest_version'] = n_not_longest
            print(f"  {key}: {n_not_longest} contested events choose a non-longest version")
        self.results['methods'][key] = entry
        self._write_json(f"selection_report_{key}.json",
                         {'method': key, 'seed': None, 'events': records})
        (self.output_dir / f'summary_{key}.txt').write_text(summary, encoding='utf-8')
        print(f"  done in {time.time()-t0:.1f}s: {evaluated['flat']}")

    def run_random(self):
        n = self.random_seeds
        print(f"\n=== random (timeline-aware, N={n} seeds) ===")
        per_seed = []
        compact_choices = {}
        for seed in range(n):
            t0 = time.time()
            strategy = self._strategy_for('random', seed=seed)
            summary, records = self.consolidator.consolidate_with_strategy(
                strategy, graph=self.graph, events=self.events, verbose=False)
            evaluated = self._evaluate(summary)
            self.random_runs.append({'seed': seed, 'summary': summary,
                                     'records': records, 'metrics': evaluated['flat']})
            per_seed.append({'seed': seed, **evaluated['flat']})
            compact_choices[str(seed)] = {
                str(r['event_id']): r['chosen_gospel']
                for r in records if not r['fallback']
            }
            print(f"  seed {seed:2d}: R-L {evaluated['flat']['rougeL_f1']:.3f} "
                  f"({time.time()-t0:.1f}s)")
        mean, std = mean_std([p for p in per_seed])
        self.results['methods']['random'] = {
            'kind': 'timeline-aware',
            'n_seeds': n,
            'seeds': list(range(n)),
            'metrics_mean': mean,
            'metrics_std': std,
            'per_seed': per_seed,
        }
        self._write_json('selection_report_random.json',
                         {'method': 'random', 'n_seeds': n, 'choices_per_seed': compact_choices})

    def run_ablation_divergence(self):
        if 'taeg' not in self.records:
            return
        report = {}
        for variant in ('taeg-no-before', 'taeg-no-same-event'):
            if variant in self.records:
                report[variant] = selection_divergence(
                    self.records['taeg'], self.records[variant])
        if report:
            print("\n=== ablation divergence vs full taeg ===")
            for variant, d in report.items():
                print(f"  {variant}: {d['n_different']}/{d['n_comparable']} choices differ")
            self._write_json('ablation_divergence.json', report)
            self.results['ablation_divergence'] = {
                v: {'n_different': d['n_different'], 'n_comparable': d['n_comparable']}
                for v, d in report.items()
            }

    def run_degradation(self):
        print(f"\n=== timeline degradation (taeg; levels {DEGRADATION_LEVELS}, "
              f"N={len(DEGRADATION_SEEDS)} seeds/level) ===")
        levels = {}
        for fraction in DEGRADATION_LEVELS:
            per_seed = []
            for seed in DEGRADATION_SEEDS:
                t0 = time.time()
                run = consolidate_degraded(fraction, seed)
                evaluated = self._evaluate(run['summary'])
                per_seed.append({'seed': seed,
                                 'n_events_kept': run['n_events_kept'],
                                 **evaluated['flat']})
                print(f"  {int(fraction*100)}% seed {seed}: "
                      f"R-L {evaluated['flat']['rougeL_f1']:.3f} ({time.time()-t0:.1f}s)")
            mean, std = mean_std(per_seed)
            levels[f"{int(fraction*100)}%"] = {
                'fraction': fraction,
                'n_seeds': len(DEGRADATION_SEEDS),
                'seeds': list(DEGRADATION_SEEDS),
                'metrics_mean': mean,
                'metrics_std': std,
                'per_seed': per_seed,
            }
        degradation = {
            'strategy': 'taeg',
            'note': ("Removed events are absent from the output: this measures "
                     "completeness/content degradation. Kendall's Tau remains 1.0 "
                     "among surviving events by construction of the timeline loop; "
                     "the reported tau uses the full-Golden-Sample heuristic matcher."),
            'levels': levels,
        }
        self.results['degradation'] = degradation
        self._write_json('results_degradation.json', degradation)

    # ---------------- reporting ----------------

    def _table_rows(self):
        """Yield (label, metrics dict or (mean, std), length) rows in paper order."""
        rows = []
        methods = self.results['methods']
        if 'lexrank' in methods:
            rows.append((f"LexRank ({LEXRANK_LENGTH} sent.)",
                         methods['lexrank']['metrics'], None))
        if 'random' in methods:
            rows.append((f"Timeline+Random (N={methods['random']['n_seeds']})",
                         (methods['random']['metrics_mean'],
                          methods['random']['metrics_std']), None))
        label_map = [
            ('priority', 'Timeline+Priority'),
            ('centroid', 'Timeline+Centroid'),
            ('longest', 'Timeline+Longest'),
            ('taeg', 'TAEG (Algorithm 1)'),
            ('taeg-no-before', 'TAEG w/o BEFORE'),
            ('taeg-no-same-event', 'TAEG w/o SAME\\_EVENT'),
        ]
        for key, label in label_map:
            if key in methods:
                rows.append((label, methods[key]['metrics'], None))
        if 'degradation' in self.results:
            for level, data in self.results['degradation']['levels'].items():
                rows.append((f"TAEG, timeline -{level} (N={data['n_seeds']})",
                             (data['metrics_mean'], data['metrics_std']), None))
        return rows

    @staticmethod
    def _fmt(value, std=None, decimals=3):
        if std is None:
            return f"{value:.{decimals}f}"
        return f"{value:.{decimals}f} ± {std:.{decimals}f}"

    def write_tables(self):
        rows = self._table_rows()
        md_lines = ["| Method | " + " | ".join(METRIC_LABELS[k] for k in METRIC_KEYS)
                    + " | Length (chars) |",
                    "|" + " :---- |" * (len(METRIC_KEYS) + 2)]
        tex_lines = ["% Auto-generated by run_experiments.py — paste into Tables 4-5",
                     "% Columns: Method & " + " & ".join(METRIC_LABELS[k] for k in METRIC_KEYS)
                     + " & Length (chars) \\\\"]
        for label, metrics, _ in rows:
            if isinstance(metrics, tuple):
                mean, std = metrics
                md_cells = [self._fmt(mean[k], std[k]) for k in METRIC_KEYS]
                tex_cells = [f"${mean[k]:.3f} \\pm {std[k]:.3f}$" for k in METRIC_KEYS]
                length_md = f"{mean['length_chars']:,.0f} ± {std['length_chars']:,.0f}"
                length_tex = f"${mean['length_chars']:.0f} \\pm {std['length_chars']:.0f}$"
            else:
                md_cells = [self._fmt(metrics[k]) for k in METRIC_KEYS]
                tex_cells = [f"{metrics[k]:.3f}" for k in METRIC_KEYS]
                length_md = f"{metrics['length_chars']:,}"
                length_tex = f"{metrics['length_chars']}"
            md_lines.append("| " + label.replace('\\_', '_') + " | "
                            + " | ".join(md_cells) + f" | {length_md} |")
            tex_lines.append(label + " & " + " & ".join(tex_cells)
                             + f" & {length_tex} \\\\")

        md = "\n".join(md_lines) + "\n"
        tex = "\n".join(tex_lines) + "\n"
        (self.output_dir / 'results_table.md').write_text(md, encoding='utf-8')
        (self.output_dir / 'results_table.tex').write_text(tex, encoding='utf-8')
        print(f"  wrote {self.output_dir / 'results_table.md'}")
        print(f"  wrote {self.output_dir / 'results_table.tex'}")
        return md

    def write_results(self):
        self.results['generated_at'] = datetime.now(timezone.utc).isoformat()
        self.results['git_commit'] = git_commit_hash()
        self.results['config'] = {
            'lexrank_length_sentences': LEXRANK_LENGTH,
            'random_seeds': self.random_seeds,
            'degradation_levels': list(DEGRADATION_LEVELS),
            'degradation_seeds_per_level': len(DEGRADATION_SEEDS),
            'golden_sample_chars': len(self.golden),
            'evaluator': {
                'rouge': 'rouge-score, use_stemmer=True',
                'bertscore': 'bert-score lang=en (roberta-large defaults)',
                'meteor': 'nltk meteor_score, lowercased word_tokenize',
                'kendall_tau': 'heuristic event matching vs Golden Sample (unchanged from published code)',
            },
        }
        self._write_json('results_all_methods.json', self.results)


def main():
    parser = argparse.ArgumentParser(
        description="JBCS revision experiment runner (see docs/JBCS_REVISION_SPEC.md)")
    parser.add_argument('--all', action='store_true',
                        help="run every method, ablations and degradation")
    parser.add_argument('--methods', type=str, default=None,
                        help=f"comma-separated subset of {ALL_METHODS}")
    parser.add_argument('--random-seeds', type=int, default=RANDOM_SEEDS_DEFAULT,
                        help="number of seeds for the random strategy (default 30)")
    parser.add_argument('--skip-degradation', action='store_true',
                        help="skip the Task 4 degradation experiment")
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()

    if not args.all and not args.methods:
        parser.error("choose --all or --methods")

    if args.all:
        methods = list(ALL_METHODS)
        degradation = not args.skip_degradation
    else:
        methods = [m.strip() for m in args.methods.split(',') if m.strip()]
        unknown = set(methods) - set(ALL_METHODS)
        if unknown:
            parser.error(f"unknown methods: {sorted(unknown)}")
        degradation = args.all and not args.skip_degradation

    t_start = time.time()
    runner = ExperimentRunner(Path(args.output_dir), args.random_seeds)

    if 'lexrank' in methods:
        runner.run_lexrank()
    for key in DETERMINISTIC_STRATEGIES:
        if key in methods:
            runner.run_deterministic_strategy(key)
    if 'random' in methods:
        runner.run_random()
    runner.run_ablation_divergence()
    if degradation:
        runner.run_degradation()

    print("\n=== writing consolidated results ===")
    runner.write_results()
    md = runner.write_tables()

    print(f"\nAll experiments finished in {(time.time()-t_start)/60:.1f} min.\n")
    print(md)


if __name__ == '__main__':
    main()
