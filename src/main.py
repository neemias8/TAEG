"""
Main orchestrator for TAEG project.

This script provides the main entry point for running the complete TAEG pipeline:
1. Data loading and validation
2. Graph construction
3. Model training (TAEG)
4. Baseline model evaluation
5. Comprehensive evaluation and comparison
6. Results analysis and visualization

Usage:
    python src/main.py --config config.yaml
    python src/main.py --mode train --epochs 50
    python src/main.py --mode evaluate --model_path outputs/best_model.pt
    python src/main.py --mode ablation

Author: Your Name
Date: September 2025
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import yaml
from dataclasses import asdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from graph_builder import TAEGGraphBuilder
from models import ModelFactory, ModelConfig, TAEGModel
from train import TAEGTrainer, TrainingConfig
from evaluate import TAEGEvaluator, EvaluationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('taeg_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TAEGPipeline:
    """Main pipeline orchestrator for TAEG project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config = self._load_config(config_path)
        self.data_loader = None
        self.graph_builder = None
        self.events = None
        self.gospel_texts = None
        self.graph_data = None
        
        # Setup directories
        self._setup_directories()
        
        logger.info("TAEG Pipeline initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'data': {
                'data_dir': 'data',
                'target_summaries_file': None  # Path to ground truth summaries
            },
            'model': {
                'hidden_dim': 256,
                'num_gat_layers': 2,
                'num_attention_heads': 8,
                'dropout_rate': 0.1,
                'text_encoder_model': 'bert-base-uncased',
                'decoder_model': 'facebook/bart-base',
                'max_input_length': 512,
                'max_output_length': 256
            },
            'training': {
                'batch_size': 4,
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'save_every_n_epochs': 5,
                'early_stopping_patience': 10,
                'output_dir': 'outputs',
                'device': 'cuda' if 'cuda' in str(os.environ.get('CUDA_VISIBLE_DEVICES', '')) else 'cpu'
            },
            'evaluation': {
                'compute_rouge': True,
                'compute_bertscore': True,
                'compute_temporal_coherence': True,
                'output_dir': 'outputs/evaluation',
                'save_results': True,
                'create_plots': True
            },
            'experiments': {
                'run_baselines': True,
                'run_ablation': True,
                'baseline_models': ['lexrank', 'pegasus', 'primera'],
                'ablation_studies': ['no_temporal_edges', 'no_graph_encoder']
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge configurations (user config overrides defaults)
            config = self._merge_configs(default_config, user_config)
        else:
            config = default_config
        
        return config
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                default[key] = self._merge_configs(default[key], value)
            else:
                default[key] = value
        return default
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        dirs_to_create = [
            self.config['training']['output_dir'],
            self.config['evaluation']['output_dir'],
            'outputs/checkpoints',
            'outputs/logs',
            'outputs/plots',
            'outputs/models'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> None:
        """Load and validate data."""
        logger.info("Loading data...")
        
        self.data_loader = DataLoader(self.config['data']['data_dir'])
        self.events, self.gospel_texts = self.data_loader.load_all_data()
        
        if not self.events:
            raise ValueError("No events loaded. Please check data files.")
        
        # Validate data
        stats = self.data_loader.validate_data()
        logger.info(f"Data validation: {stats}")
        
        # Save data statistics
        with open('outputs/data_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Loaded {len(self.events)} events from {len(self.gospel_texts)} gospels")
    
    def build_graph(self, include_temporal_edges: bool = True) -> None:
        """Build TAEG graph."""
        logger.info(f"Building graph (temporal_edges={include_temporal_edges})...")
        
        if self.data_loader is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.graph_builder = TAEGGraphBuilder(self.data_loader)
        self.graph_data = self.graph_builder.build_graph(include_temporal_edges)
        
        # Get and save graph statistics
        stats = self.graph_builder.get_graph_statistics()
        logger.info(f"Graph statistics: {asdict(stats)}")
        
        with open('outputs/graph_statistics.json', 'w') as f:
            json.dump(asdict(stats), f, indent=2)
        
        # Create graph visualization
        try:
            self.graph_builder.visualize_graph_structure('outputs/plots/graph_structure.png')
        except Exception as e:
            logger.warning(f"Could not create graph visualization: {e}")
        
        # Export graph
        self.graph_builder.export_graph('outputs/taeg_graph.graphml')
    
    def load_target_summaries(self) -> List[str]:
        """Load target summaries for training/evaluation."""
        summaries_file = self.config['data'].get('target_summaries_file')
        
        if summaries_file and Path(summaries_file).exists():
            with open(summaries_file, 'r') as f:
                if summaries_file.endswith('.json'):
                    summaries_data = json.load(f)
                    # Assuming format: {"event_id": "summary", ...}
                    summaries = [
                        summaries_data.get(event.event_id, f"Summary for {event.event_id}")
                        for event in self.events
                    ]
                else:
                    # Assuming one summary per line
                    summaries = [line.strip() for line in f.readlines()]
        else:
            summaries = [
                f"Generated summary for event {event.event_id} covering the narrative from {', '.join(event.get_all_gospels_with_refs())}"
                for event in self.events
            ]
        
        return summaries
    
    def train_taeg_model(self) -> TAEGModel:
        """Train the TAEG model."""
        logger.info("Training TAEG model...")
        
        if self.graph_data is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        # Create model and training configurations
        model_config = ModelConfig(**self.config['model'])
        training_config = TrainingConfig(**self.config['training'])
        
        # Load target summaries
        target_summaries = self.load_target_summaries()
        
        # Initialize trainer
        trainer = TAEGTrainer(model_config, training_config)
        
        # Prepare data
        trainer.prepare_data(self.data_loader, target_summaries)
        trainer.prepare_training()
        
        # Train model
        training_history = trainer.train()
        
        # Save training history
        with open('outputs/training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save model configuration
        with open('outputs/model_config.json', 'w') as f:
            json.dump(asdict(model_config), f, indent=2)
        
        logger.info("TAEG model training completed")
        return trainer.model
    
    def evaluate_models(self, taeg_model: Optional[TAEGModel] = None) -> Dict[str, Any]:
        """Evaluate all models and compare results."""
        logger.info("Starting model evaluation...")
        
        # Create evaluation config
        eval_config = EvaluationConfig(**self.config['evaluation'])
        evaluator = TAEGEvaluator(eval_config, self.data_loader)
        
        # Load target summaries
        target_summaries = self.load_target_summaries()
        
        # Prepare test data
        class TestData:
            def __init__(self, events, graph_data):
                self.events = events
                self.graph_data = graph_data
        
        test_data = TestData(self.events, self.graph_data)
        
        # Evaluate models
        all_results = []
        
        # Evaluate TAEG model (if provided)
        if taeg_model is not None:
            try:
                taeg_results = evaluator.evaluate_model(taeg_model, 'TAEG', test_data, target_summaries)
                all_results.append(taeg_results)
            except Exception as e:
                logger.error(f"Error evaluating TAEG model: {e}")
        
        # Evaluate baseline models
        if self.config['experiments']['run_baselines']:
            baseline_models = self.config['experiments']['baseline_models']
            
            for baseline_name in baseline_models:
                try:
                    logger.info(f"Evaluating {baseline_name} baseline...")
                    baseline_model = ModelFactory.create_model(baseline_name)
                    baseline_results = evaluator.evaluate_model(
                        baseline_model, baseline_name.upper(), test_data, target_summaries
                    )
                    all_results.append(baseline_results)
                except Exception as e:
                    logger.error(f"Error evaluating {baseline_name}: {e}")
        
        # Compare models
        if all_results:
            comparison = evaluator.compare_models(all_results)
            
            # Create plots
            evaluator.create_evaluation_plots(all_results, comparison)
            
            # Save results
            evaluator.save_results(all_results, comparison)
            
            logger.info("Model evaluation completed")
            return {
                'individual_results': all_results,
                'comparison': comparison
            }
        else:
            logger.warning("No models evaluated successfully")
            return {}
    
    def run_ablation_studies(self) -> Dict[str, Any]:
        """Run ablation studies."""
        logger.info("Running ablation studies...")
        
        ablation_results = {}
        
        if 'no_temporal_edges' in self.config['experiments']['ablation_studies']:
            logger.info("Ablation: TAEG without temporal edges")
            
            try:
                # Build graph without temporal edges
                self.build_graph(include_temporal_edges=False)
                
                # Train model
                ablation_model = self.train_taeg_model()
                
                # Evaluate
                eval_results = self.evaluate_models(ablation_model)
                ablation_results['no_temporal_edges'] = eval_results
                
                # Restore normal graph
                self.build_graph(include_temporal_edges=True)
                
            except Exception as e:
                logger.error(f"Error in ablation study: {e}")
        
        return ablation_results
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete TAEG pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE TAEG PIPELINE")
        logger.info("=" * 80)
        
        start_time = time.time()
        results = {}
        
        try:
            # Step 1: Load data
            self.load_data()
            results['data_loaded'] = True
            
            # Step 2: Build graph
            self.build_graph()
            results['graph_built'] = True
            
            # Step 3: Train TAEG model
            taeg_model = self.train_taeg_model()
            results['model_trained'] = True
            
            # Step 4: Evaluate models
            evaluation_results = self.evaluate_models(taeg_model)
            results['evaluation_results'] = evaluation_results
            
            # Step 5: Run ablation studies
            if self.config['experiments']['run_ablation']:
                ablation_results = self.run_ablation_studies()
                results['ablation_results'] = ablation_results
            
            # Step 6: Generate final report
            self._generate_final_report(results)
            
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            results['total_time'] = total_time
            results['success'] = True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _generate_final_report(self, results: Dict[str, Any]) -> None:
        """Generate final report with all results."""
        logger.info("Generating final report...")
        
        report = {
            'pipeline_summary': {
                'data_events': len(self.events) if self.events else 0,
                'graph_nodes': self.graph_data.num_nodes if self.graph_data else 0,
                'graph_edges': self.graph_data.num_edges if self.graph_data else 0
            },
            'model_results': results.get('evaluation_results', {}),
            'ablation_results': results.get('ablation_results', {}),
            'pipeline_time': results.get('total_time', 0)
        }
        
        # Save complete report
        with open('outputs/final_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        logger.info("Final report saved to outputs/final_report.json")
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> None:
        """Generate human-readable markdown report."""
        md_content = [
            "# TAEG Project Results\\n",
            f"**Pipeline completed at:** {time.strftime('%Y-%m-%d %H:%M:%S')}\\n",
            f"**Total execution time:** {report['pipeline_time']:.2f} seconds\\n\\n",
            
            "## Data Summary\\n",
            f"- **Events processed:** {report['pipeline_summary']['data_events']}\\n",
            f"- **Graph nodes:** {report['pipeline_summary']['graph_nodes']}\\n",
            f"- **Graph edges:** {report['pipeline_summary']['graph_edges']}\\n\\n",
            
            "## Model Performance\\n"
        ]
        
        # Add model results if available
        eval_results = report.get('model_results', {})
        if 'comparison' in eval_results:
            comparison = eval_results['comparison']
            md_content.append("### Best Performing Models\\n")
            
            for metric, data in comparison.get('metrics_comparison', {}).items():
                best_model = data.get('best_model', 'N/A')
                md_content.append(f"- **{metric}:** {best_model}\\n")
        
        md_content.append("\\n## Files Generated\\n")
        md_content.append("- `outputs/final_report.json` - Complete results\\n")
        md_content.append("- `outputs/data_statistics.json` - Data validation stats\\n")
        md_content.append("- `outputs/graph_statistics.json` - Graph analysis\\n")
        md_content.append("- `outputs/training_history.json` - Training metrics\\n")
        md_content.append("- `outputs/evaluation/` - Detailed evaluation results\\n")
        md_content.append("- `outputs/plots/` - Visualization plots\\n")
        
        with open('outputs/RESULTS.md', 'w') as f:
            f.writelines(md_content)
        
        logger.info("Markdown report saved to outputs/RESULTS.md")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description='TAEG Project Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'complete', 'ablation'], 
                       default='complete', help='Pipeline mode')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = TAEGPipeline(args.config)
        
        # Override config with command line arguments
        if args.data_dir:
            pipeline.config['data']['data_dir'] = args.data_dir
        if args.epochs:
            pipeline.config['training']['num_epochs'] = args.epochs
        if args.batch_size:
            pipeline.config['training']['batch_size'] = args.batch_size
        if args.output_dir:
            pipeline.config['training']['output_dir'] = args.output_dir
            pipeline.config['evaluation']['output_dir'] = f"{args.output_dir}/evaluation"
        
        # Run based on mode
        if args.mode == 'complete':
            results = pipeline.run_complete_pipeline()
            
        elif args.mode == 'train':
            pipeline.load_data()
            pipeline.build_graph()
            taeg_model = pipeline.train_taeg_model()
            results = {'model_trained': True}
            
        elif args.mode == 'evaluate':
            pipeline.load_data()
            pipeline.build_graph()
            
            # Load model if path provided
            taeg_model = None
            if args.model_path and Path(args.model_path).exists():
                from models import ModelConfig
                model_config = ModelConfig()
                taeg_model = TAEGModel(model_config)
                taeg_model.load_model(args.model_path)
            
            results = pipeline.evaluate_models(taeg_model)
            
        elif args.mode == 'ablation':
            pipeline.load_data()
            results = pipeline.run_ablation_studies()
        
        if results.get('success', True):
            print("\\n" + "=" * 50)
            print("TAEG PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"Results saved to: {args.output_dir}")
            print("Check outputs/RESULTS.md for a summary")
        else:
            print("\\nPipeline failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        print(f"\\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
