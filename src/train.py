"""
Training module for TAEG project.

This module implements the training pipeline for the TAEG model, including:
- Data preparation and batching
- Training loop with validation
- Learning rate scheduling
- Model checkpointing
- Tensorboard logging

Author: Your Name
Date: September 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

from .data_loader import DataLoader, Event
from .graph_builder import TAEGGraphBuilder
from .models import TAEGModel, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Data settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Scheduler settings
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Checkpointing
    save_every_n_epochs: int = 5
    save_best_model: bool = True
    early_stopping_patience: int = 10
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1


class TAEGDataset(Dataset):
    """Dataset class for TAEG training."""
    
    def __init__(self, events: List[Event], graph_data: Data, 
                 target_summaries: List[str], data_loader: DataLoader):
        """
        Initialize dataset.
        
        Args:
            events: List of events
            graph_data: Graph data structure
            target_summaries: Ground truth summaries
            data_loader: DataLoader instance for text extraction
        """
        self.events = events
        self.graph_data = graph_data
        self.target_summaries = target_summaries
        self.data_loader = data_loader
        
        assert len(events) == len(target_summaries), "Mismatch in events and summaries count"
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get single training example."""
        event = self.events[idx]
        target_summary = self.target_summaries[idx]
        
        # Get text content for this event
        text_content = self.data_loader.get_concatenated_text_for_event(event)
        
        return {
            'event_id': event.event_id,
            'text_content': text_content,
            'target_summary': target_summary,
            'event_idx': idx
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching graph data."""
    # Extract components
    event_ids = [item['event_id'] for item in batch]
    text_contents = [item['text_content'] for item in batch]
    target_summaries = [item['target_summary'] for item in batch]
    event_indices = [item['event_idx'] for item in batch]
    
    return {
        'event_ids': event_ids,
        'text_contents': text_contents,
        'target_summaries': target_summaries,
        'event_indices': event_indices
    }


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return False
        
        if self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            return False
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


class TAEGTrainer:
    """Main trainer class for TAEG model."""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        
        # Setup directories
        self._setup_directories()
        
        # Initialize model
        self.model = TAEGModel(model_config)
        self.model.to(training_config.device)
        
        # Training components (will be initialized in prepare_training)
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
        # Data components
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = defaultdict(list)
        
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        for dir_path in [self.training_config.output_dir, 
                        self.training_config.checkpoint_dir,
                        self.training_config.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, data_loader: DataLoader, target_summaries: List[str]) -> None:
        """
        Prepare training data.
        
        Args:
            data_loader: DataLoader with events and texts
            target_summaries: List of target summaries for each event
        """
        logger.info("Preparing training data...")
        
        # Build graph
        graph_builder = TAEGGraphBuilder(data_loader)
        graph_data = graph_builder.build_graph(include_temporal_edges=True)
        
        # Create dataset
        events = data_loader.events
        dataset = TAEGDataset(events, graph_data, target_summaries, data_loader)
        
        # Split data
        train_size = int(self.training_config.train_split * len(dataset))
        val_size = int(self.training_config.val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = TorchDataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        self.val_loader = TorchDataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        self.test_loader = TorchDataLoader(
            test_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Store graph data for use during training
        self.graph_data = graph_data
        
        logger.info(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    def prepare_training(self) -> None:
        """Initialize training components."""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Scheduler
        if self.training_config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.training_config.num_epochs
            )
        elif self.training_config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=self.training_config.scheduler_factor
            )
        elif self.training_config.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', 
                patience=self.training_config.scheduler_patience,
                factor=self.training_config.scheduler_factor
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.training_config.early_stopping_patience,
            mode='min'
        )
        
        logger.info("Training components initialized")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Create graph data for this batch
            batch_graph_data, batch_indices = self._create_batch_graph_data(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_graph_data, batch_indices, batch['target_summaries'])
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.training_config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Logging
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            if batch_idx % self.training_config.log_every_n_steps == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                           f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return {'train_loss': np.mean(epoch_losses)}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Create graph data for this batch
                batch_graph_data, batch_indices = self._create_batch_graph_data(batch)

                # Forward pass
                outputs = self.model(batch_graph_data, batch_indices, batch['target_summaries'])
                loss = outputs['loss']

                val_losses.append(loss.item())
        
        return {'val_loss': np.mean(val_losses)}
    
    def _create_batch_graph_data(self, batch: Dict[str, Any]) -> Tuple[Data, torch.Tensor]:
        """Return the shared graph data and batch node indices."""
        if self.graph_data is None:
            raise ValueError("Graph data has not been initialised. Call prepare_data first.")

        device = torch.device(self.training_config.device)
        batch_graph_data = self.graph_data.to(device)
        batch_indices = torch.tensor(batch["event_indices"], dtype=torch.long, device=device)

        return batch_graph_data, batch_indices

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_history': dict(self.training_history),
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.training_config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.training_config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.training_config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history dictionary
        """
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        logger.info(f"Starting training for {self.training_config.num_epochs} epochs")
        logger.info(f"Device: {self.training_config.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.training_config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.training_config.eval_every_n_epochs == 0:
                val_metrics = self.validate_epoch()
                
                # Update learning rate scheduler
                if self.scheduler:
                    if self.training_config.scheduler_type == "plateau":
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                
                # Track metrics
                for key, value in {**train_metrics, **val_metrics}.items():
                    self.training_history[key].append(value)
                
                # Check for best model
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                
                # Save checkpoint
                if (epoch + 1) % self.training_config.save_every_n_epochs == 0 or is_best:
                    self.save_checkpoint(epoch + 1, is_best)
                
                # Early stopping
                if self.early_stopping(val_metrics['val_loss'], self.model):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Logging
                logger.info(f"Epoch {epoch + 1}/{self.training_config.num_epochs} - "
                           f"Train Loss: {train_metrics['train_loss']:.4f}, "
                           f"Val Loss: {val_metrics['val_loss']:.4f}, "
                           f"Best Val Loss: {self.best_val_loss:.4f}")
            else:
                # Only track training metrics
                for key, value in train_metrics.items():
                    self.training_history[key].append(value)
                
                logger.info(f"Epoch {epoch + 1}/{self.training_config.num_epochs} - "
                           f"Train Loss: {train_metrics['train_loss']:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        self.save_checkpoint(self.current_epoch + 1, False)
        
        return dict(self.training_history)
    
    def evaluate_on_test(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        test_losses = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch_graph_data, batch_indices = self._create_batch_graph_data(batch)
                outputs = self.model(batch_graph_data, batch_indices, batch['target_summaries'])
                test_losses.append(outputs['loss'].item())
        
        test_metrics = {'test_loss': np.mean(test_losses)}
        logger.info(f"Test metrics: {test_metrics}")
        
        return test_metrics


def main():
    """Example training script."""
    # Configuration
    model_config = ModelConfig(
        hidden_dim=128,
        num_gat_layers=2,
        max_input_length=256,
        max_output_length=128
    )
    
    training_config = TrainingConfig(
        batch_size=2,
        num_epochs=10,
        learning_rate=0.001,
        save_every_n_epochs=2
    )
    
    # Load data (placeholder - replace with actual data loading)
    data_loader = DataLoader("data")
    events, gospel_texts = data_loader.load_all_data()
    
    if not events:
        logger.error("No events loaded. Please check data files.")
        return
    
    # Create dummy target summaries (replace with actual ground truth)
    target_summaries = [f"Summary for event {event.event_id}" for event in events]
    
    # Initialize trainer
    trainer = TAEGTrainer(model_config, training_config)
    
    # Prepare data and training
    trainer.prepare_data(data_loader, target_summaries)
    trainer.prepare_training()
    
    # Train model
    history = trainer.train()
    
    # Evaluate
    test_metrics = trainer.evaluate_on_test()
    
    print("Training completed!")
    print(f"Final test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
