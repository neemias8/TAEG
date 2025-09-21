"""
Models module for TAEG project.

This module implements:
1. TAEG model: GAT-based graph neural network + Transformer decoder
2. Baseline models: PEGASUS, PRIMERA, LexRank
3. Model utilities and training components

Author: Your Name  
Date: September 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM,
    PegasusForConditionalGeneration, PegasusTokenizer,
    GenerationConfig
)
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for TAEG model."""
    # Graph encoder settings
    hidden_dim: int = 256
    num_gat_layers: int = 2
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # Text encoder settings
    text_encoder_model: str = "bert-base-uncased"
    max_input_length: int = 512
    
    # Decoder settings
    decoder_model: str = "facebook/bart-base"
    max_output_length: int = 256
    
    # Training settings
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def generate_summary(self, input_data: Any) -> str:
        """Generate summary from input data."""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model to file."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model from file."""
        pass


class GraphAttentionEncoder(nn.Module):
    """Graph Attention Network encoder for processing event graphs."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.gat_layers = nn.ModuleList()
        input_dim = 768  # BERT base hidden size

        for layer_idx in range(config.num_gat_layers):
            in_channels = input_dim if layer_idx == 0 else config.hidden_dim
            out_channels = config.hidden_dim
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=config.num_attention_heads,
                    dropout=config.dropout_rate,
                    concat=layer_idx < config.num_gat_layers - 1,
                )
            )

        final_dim = (
            config.hidden_dim * config.num_attention_heads
            if config.num_gat_layers == 1
            else config.hidden_dim
        )
        self.graph_projection = nn.Linear(final_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings shaped [num_nodes, hidden_dim]."""
        for layer_idx, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if layer_idx < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)

        x = self.graph_projection(x)
        x = self.dropout(F.relu(x))
        return x


class TAEGModel(BaseModel, nn.Module):
    """Temporal Alignment Event Graph model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_model)
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)

        self.graph_encoder = GraphAttentionEncoder(config)

        self.decoder_model = AutoModelForSeq2SeqLM.from_pretrained(config.decoder_model)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(config.decoder_model)

        self.graph_to_decoder = nn.Linear(config.hidden_dim, self.decoder_model.config.d_model)

    def _model_device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode raw texts into fixed-size embeddings using the text encoder."""
        device = next(self.text_encoder.parameters()).device
        tokens = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings

    def _get_text_embeddings(self, graph_data: Data, device: torch.device) -> torch.Tensor:
        cached = getattr(graph_data, 'cached_text_embeddings', None)
        if cached is not None:
            if cached.device != device:
                cached = cached.to(device)
                graph_data.cached_text_embeddings = cached
            return cached

        if not hasattr(graph_data, 'text_contents'):
            raise ValueError('Graph data must include text_contents for each node.')

        embeddings = self.encode_texts(graph_data.text_contents).to(device)
        graph_data.cached_text_embeddings = embeddings
        return embeddings

    def forward(
        self,
        graph_data: Data,
        node_indices: torch.Tensor,
        target_texts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the full TAEG model and optionally compute training loss."""
        device = self._model_device()
        graph_data = graph_data.to(device)

        if not isinstance(node_indices, torch.Tensor):
            node_indices = torch.tensor(node_indices, dtype=torch.long, device=device)
        else:
            node_indices = node_indices.to(device)
        node_indices = node_indices.view(-1)

        text_embeddings = self._get_text_embeddings(graph_data, device)
        node_embeddings = self.graph_encoder(text_embeddings, graph_data.edge_index)
        selected_embeddings = node_embeddings.index_select(0, node_indices)
        decoder_hidden = self.graph_to_decoder(selected_embeddings)

        encoder_outputs = BaseModelOutput(last_hidden_state=decoder_hidden.unsqueeze(1))
        encoder_attention_mask = torch.ones(
            decoder_hidden.size(0),
            1,
            dtype=torch.long,
            device=device,
        )

        if target_texts is not None:
            target_tokens = self.decoder_tokenizer(
                target_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_output_length,
                return_tensors="pt",
            ).to(device)

            outputs = self.decoder_model(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=target_tokens['input_ids'],
                decoder_attention_mask=target_tokens['attention_mask'],
                labels=target_tokens['input_ids'],
            )

            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'node_embeddings': node_embeddings,
                'selected_embeddings': selected_embeddings,
                'decoder_hidden': decoder_hidden,
            }

        generated_ids = self.decoder_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=self.config.max_output_length,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
        )

        return {
            'generated_ids': generated_ids,
            'node_embeddings': node_embeddings,
            'selected_embeddings': selected_embeddings,
            'decoder_hidden': decoder_hidden,
        }

    def generate_summary(self, input_data: Any) -> str:
        """Generate a summary for a given event."""
        if isinstance(input_data, tuple):
            graph_data, event_index = input_data
        elif isinstance(input_data, dict):
            graph_data = input_data.get('graph_data')
            event_index = input_data.get('event_index', 0)
        else:
            graph_data = input_data
            event_index = 0

        device = self._model_device()
        indices = torch.tensor([event_index], dtype=torch.long, device=device)

        self.eval()
        with torch.no_grad():
            outputs = self.forward(graph_data, indices, target_texts=None)
            summary = self.decoder_tokenizer.decode(
                outputs['generated_ids'][0],
                skip_special_tokens=True,
            ).strip()
        return summary

    def save_model(self, path: str) -> None:
        """Save model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"TAEG model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model state."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"TAEG model loaded from {path}")


class PegasusBaseline(BaseModel):
    """PEGASUS baseline model for abstractive summarization."""
    
    def __init__(self, model_name: str = "google/pegasus-xsum"):
        self.model_name = model_name
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        
        # Set generation config
        self.generation_config = GenerationConfig(
            max_length=256,
            min_length=50,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )
    
    def generate_summary(self, input_text: str) -> str:
        """Generate summary from concatenated input text."""
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                generation_config=self.generation_config
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def save_model(self, path: str) -> None:
        """Save model (PEGASUS is pre-trained, so just save config)."""
        torch.save({'model_name': self.model_name}, path)
    
    def load_model(self, path: str) -> None:
        """Load model config."""
        checkpoint = torch.load(path)
        self.model_name = checkpoint['model_name']


class PrimeraBaseline(BaseModel):
    """PRIMERA baseline model for multi-document summarization."""
    
    def __init__(self, model_name: str = "allenai/PRIMERA"):
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except:
            # Fallback to BART if PRIMERA not available
            logger.warning("PRIMERA not available, using BART instead")
            self.model_name = "facebook/bart-large-cnn"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def generate_summary(self, input_texts: List[str]) -> str:
        """Generate summary from multiple input texts."""
        # Concatenate input texts with special separator
        combined_text = " <doc-sep> ".join(input_texts)
        
        # Tokenize
        inputs = self.tokenizer(
            combined_text,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,
                min_length=50,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def save_model(self, path: str) -> None:
        """Save model config."""
        torch.save({'model_name': self.model_name}, path)
    
    def load_model(self, path: str) -> None:
        """Load model config."""
        checkpoint = torch.load(path)
        self.model_name = checkpoint['model_name']


class LexRankBaseline(BaseModel):
    """LexRank baseline for extractive summarization."""
    
    def __init__(self, num_sentences: int = 5, threshold: float = 0.1):
        self.num_sentences = num_sentences
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def _sentence_similarity(self, sentences: List[str]) -> np.ndarray:
        """Compute sentence similarity matrix using TF-IDF and cosine similarity."""
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except ValueError:
            # Handle empty vocabulary
            return np.zeros((len(sentences), len(sentences)))
    
    def _lexrank_scores(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Compute LexRank scores using PageRank algorithm."""
        # Create adjacency matrix based on threshold
        adj_matrix = similarity_matrix > self.threshold
        
        # Convert to NetworkX graph
        graph = nx.from_numpy_array(adj_matrix.astype(float))
        
        if len(graph.nodes()) == 0:
            return np.zeros(similarity_matrix.shape[0])
        
        try:
            # Compute PageRank scores
            pagerank_scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
            scores = np.array([pagerank_scores.get(i, 0) for i in range(len(graph.nodes()))])
            return scores
        except:
            # Fallback to uniform scores
            return np.ones(similarity_matrix.shape[0]) / similarity_matrix.shape[0]
    
    def generate_summary(self, input_text: str) -> str:
        """Generate extractive summary using LexRank."""
        # Split into sentences (simple approach)
        sentences = [s.strip() for s in input_text.split('.') if s.strip()]
        
        if len(sentences) <= self.num_sentences:
            return input_text
        
        # Compute similarity matrix
        similarity_matrix = self._sentence_similarity(sentences)
        
        # Compute LexRank scores
        scores = self._lexrank_scores(similarity_matrix)
        
        # Select top sentences
        top_indices = np.argsort(scores)[-self.num_sentences:]
        top_indices = sorted(top_indices)  # Maintain original order
        
        # Generate summary
        summary_sentences = [sentences[i] for i in top_indices]
        summary = '. '.join(summary_sentences) + '.'
        
        return summary
    
    def save_model(self, path: str) -> None:
        """Save LexRank parameters."""
        torch.save({
            'num_sentences': self.num_sentences,
            'threshold': self.threshold
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load LexRank parameters."""
        checkpoint = torch.load(path)
        self.num_sentences = checkpoint['num_sentences']
        self.threshold = checkpoint['threshold']


class ModelFactory:
    """Factory class for creating models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """
        Create model instance.
        
        Args:
            model_type: Type of model ('taeg', 'pegasus', 'primera', 'lexrank')
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model instance
        """
        if model_type.lower() == 'taeg':
            config = kwargs.get('config', ModelConfig())
            return TAEGModel(config)
        elif model_type.lower() == 'pegasus':
            model_name = kwargs.get('model_name', 'google/pegasus-xsum')
            return PegasusBaseline(model_name)
        elif model_type.lower() == 'primera':
            model_name = kwargs.get('model_name', 'allenai/PRIMERA')
            return PrimeraBaseline(model_name)
        elif model_type.lower() == 'lexrank':
            num_sentences = kwargs.get('num_sentences', 5)
            threshold = kwargs.get('threshold', 0.1)
            return LexRankBaseline(num_sentences, threshold)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Example usage of models."""
    # Create model config
    config = ModelConfig(
        hidden_dim=128,
        num_gat_layers=2,
        max_input_length=256,
        max_output_length=128
    )
    
    # Create models
    taeg_model = ModelFactory.create_model('taeg', config=config)
    pegasus_model = ModelFactory.create_model('pegasus')
    primera_model = ModelFactory.create_model('primera')
    lexrank_model = ModelFactory.create_model('lexrank', num_sentences=3)
    
    print("Models created successfully!")
    
    # Example text
    sample_text = "This is a sample text for summarization. It contains multiple sentences. The models should be able to process this text and generate summaries."
    
    # Test baselines
    print("\nTesting baseline models:")
    
    try:
        pegasus_summary = pegasus_model.generate_summary(sample_text)
        print(f"PEGASUS: {pegasus_summary}")
    except Exception as e:
        print(f"PEGASUS error: {e}")
    
    try:
        primera_summary = primera_model.generate_summary([sample_text])
        print(f"PRIMERA: {primera_summary}")
    except Exception as e:
        print(f"PRIMERA error: {e}")
    
    try:
        lexrank_summary = lexrank_model.generate_summary(sample_text)
        print(f"LexRank: {lexrank_summary}")
    except Exception as e:
        print(f"LexRank error: {e}")


if __name__ == "__main__":
    main()
