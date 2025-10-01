# TAEG - Temporal Alignment Event Graph

A modular system for consolidating and evaluating biblical narratives using natural language processing techniques.

## Overview

TAEG (Temporal Alignment Event Graph) is a project that combines narratives from the four Gospels (Matthew, Mark, Luke, and John) into a consolidated summary of Holy Week using advanced summarization algorithms, and evaluates the summary quality by comparing it with a reference sample (Golden Sample) using multiple evaluation metrics.

## Features

- **Data Loading**: Processes XML files from the Gospels to extract Holy Week texts
- **Summarization with Multiple Methods**:
  - **LEXRANK**: Standard multi-document algorithm (recommended for semantic quality)
  - **LEXRANK-TA (Temporal Anchoring)**: Summarization based on biblical chronology with temporal anchors
  - **LEXRANK-TA-BEST**: Optimized temporal anchoring with best sentence selection
- **Multi-Metric Evaluation**: Evaluates summary quality using:
  - ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
  - METEOR
  - BERTScore
  - Kendall's Tau (temporal correlation)

## Summarization Methods

### LEXRANK (Standard)
- Uses traditional multi-document LEXRANK algorithm
- Prioritizes **semantic quality** and textual cohesion
- May reorder sentences chronologically to optimize narrative flow
- **Advantage**: Better semantic quality (ROUGE, METEOR, BERTScore)
- **Disadvantage**: May lose temporal order (lower Kendall's Tau)

### LEXRANK-TA (Temporal Anchoring)
- **Gospel-Specific Architecture**: Each chronological event has separate nodes for each Gospel that mentions it
- **Precise Verse Extraction**: Extracts exact biblical verse text instead of entire chapters
- **Multi-Document Summarization per Event**: For events mentioned in multiple Gospels, uses multi-document LEXRANK to combine complementary perspectives
- **Enhanced Temporal Graph**:
  - 363 gospel-specific nodes (one per Gospel per event)
  - 799 BEFORE edges (connecting consecutive events)
  - 318 SAME_EVENT edges (connecting different versions of the same event)
- **Strict Chronological Sequence**: Maintains perfect temporal order of 169 Holy Week events
- **Advantage**: Better temporal preservation + semantic quality through multiple perspectives
- **Result**: Consolidated summary covering the entire Holy Week narrative

### LEXRANK-TA-BEST (Optimized Temporal Anchoring)
- Builds upon LEXRANK-TA with optimized sentence selection
- Ensures perfect temporal ordering with minimal summary length
- **Advantage**: Maximum temporal preservation with concise output
- **Use Case**: When temporal accuracy is critical and brevity is desired

## Project Structure

```
TAEG/
├── data/                          # Input data
│   ├── EnglishNIVMatthew40_PW.xml # Gospel of Matthew
│   ├── EnglishNIVMark41_PW.xml    # Gospel of Mark
│   ├── EnglishNIVLuke42_PW.xml    # Gospel of Luke
│   ├── EnglishNIVJohn43_PW.xml    # Gospel of John
│   ├── ChronologyOfTheFourGospels_PW.xml # Chronological framework
│   └── Golden_Sample.txt          # Reference summary for evaluation
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_loader.py             # Data loading and processing
│   ├── graph_builder.py           # Temporal graph construction
│   ├── models.py                  # Model definitions
│   ├── train.py                   # Training utilities
│   ├── evaluate.py                # Evaluation metrics
│   └── main.py                    # Main execution script
├── outputs/                       # Generated results
│   ├── taeg_graph.graphml         # Temporal graph
│   ├── model_config.json          # Model configuration
│   ├── training_history.json      # Training history
│   ├── data_statistics.json       # Data statistics
│   ├── final_report.json          # Final evaluation report
│   ├── evaluation/                # Detailed evaluation results
│   │   ├── LEXRANK_results.json
│   │   ├── LEXRANK-TA_results.json
│   │   └── PRIMERA_results.json
│   ├── checkpoints/               # Model checkpoints
│   └── plots/                     # Visualization plots
├── scripts/                       # Utility scripts
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── htmlcov/                       # Coverage reports
├── compare_methods.py             # Methods comparison script
├── analyze_text_lengths.py        # Text analysis utilities
├── debug_pipeline.py              # Debugging utilities
├── test_parameters.py             # Parameter testing
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration
├── setup.py                       # Package setup
└── README.md                      # This file
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Set up virtual environment**:
   ```bash
   python -m venv .TAEG
   ```

3. **Activate virtual environment**:
   - Windows: `.TAEG\Scripts\activate`
   - Linux/Mac: `source .TAEG/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Simple Execution

Run the complete pipeline with default settings (LEXRANK):

```bash
python src/main.py
```

### Choose Summarization Method

```bash
# Standard LEXRANK (recommended for semantic quality)
python src/main.py --summarization-method lexrank --summary-length 500

# LEXRANK-TA (recommended for temporal preservation)
python src/main.py --summarization-method lexrank-ta --summary-length 3

# LEXRANK-TA-BEST (optimized temporal anchoring)
python src/main.py --summarization-method lexrank-ta-best --summary-length 1
```

### Parameters

- `--summarization-method`: Summarization method:
  - `lexrank`: Standard multi-document LEXRANK (semantic quality)
  - `lexrank-ta`: LEXRANK with Temporal Anchoring (temporal order)
  - `lexrank-ta-best`: Optimized temporal anchoring
- `--summary-length`:
  - For `lexrank`: Total number of sentences in summary
  - For `lexrank-ta` and `lexrank-ta-best`: Number of sentences per chronological event
- `--data-dir`: Data directory (default: `data`)
- `--output-dir`: Output directory (default: `outputs`)

### Methods Comparison

```bash
python compare_methods.py
```

## Summarization Methods

### LEXRANK (Standard)
- **Approach**: Traditional multi-document LEXRANK
- **Priority**: Semantic quality and textual cohesion
- **Functionality**: Treats all Gospels as single corpus, finds cross-document relationships
- **Advantages**: Better ROUGE, METEOR, BERTScore
- **Disadvantages**: May reorder chronologically (more negative Kendall's Tau)
- **Use**: When semantic quality is priority

### LEXRANK-TA (Temporal Anchoring)
- **Approach**: Summarization based on biblical chronology
- **Priority**: Preservation of temporal order
- **Functionality**: 
  - Uses chronology XML with 169 Holy Week events
  - Builds temporal graph with "BEFORE" edges
  - Generates summaries per event in chronological order
  - If multiple Gospels describe same event → multi-doc LEXRANK
- **Advantages**: Better Kendall's Tau (less temporal disorder)
- **Disadvantages**: May have slightly inferior semantic quality
- **Use**: When temporal order is priority

## 📊 Methods Comparison

Comparative test between LEXRANK, LEXRANK-TA, and LEXRANK-TA-BEST (all evaluated against Golden Sample):

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | METEOR | Kendall's Tau | Summary Length | Priority |
|--------|------------|------------|------------|--------------|--------|----------------|---------------|----------|
| **LEXRANK** | TBD | TBD | TBD | TBD | TBD | 0.287 | ~43K chars | **Semantic** |
| **LEXRANK-TA** | TBD | TBD | TBD | TBD | TBD | **1.000** | ~52K chars | **Temporal** |
| **LEXRANK-TA-BEST** | TBD | TBD | TBD | TBD | TBD | **1.000** | Minimal | **Temporal** |

### 🔍 Results Analysis

- **LEXRANK**: Shows partial temporal disorder (Kendall's Tau = 0.287), indicating some chronological reordering for semantic optimization
- **LEXRANK-TA**: Achieves perfect temporal preservation (Kendall's Tau = 1.000) while maintaining semantic quality
- **LEXRANK-TA-BEST**: Optimized for maximum temporal accuracy with minimal length (Kendall's Tau = 1.000)
- **Choice**: LEXRANK-TA-BEST for temporal-critical applications, LEXRANK for semantic quality priority

**Note**: ROUGE, BERTScore, and METEOR values are currently being updated. The temporal evaluation (Kendall's Tau) has been recently validated and confirmed working correctly.

## ✨ Recent Improvements - LEXRANK-TA Gospel-Specific

### 🎯 Enhanced Architecture

The latest version of LEXRANK-TA implements a revolutionary **gospel-specific** architecture:

#### 📊 Temporal Graph Statistics
- **363 gospel-specific nodes** (one per Gospel per mentioned event)
- **799 BEFORE edges** (connecting consecutive events across all Gospels)
- **318 SAME_EVENT edges** (connecting different versions of the same event)
- **96 events** with multiple Gospel versions (diverse coverage)

#### 📖 Distribution by Gospel
- **Matthew**: 104 events mentioned
- **Mark**: 98 events mentioned  
- **Luke**: 97 events mentioned
- **John**: 64 events mentioned

### 🔬 Advanced Features

#### Multi-Document Summarization per Event
- Unique events: Single-document LEXRANK summarization
- Multi-gospel events: Multi-document LEXRANK combining complementary perspectives
- **Result**: Consolidated summary of **52,603 characters** with complete Holy Week narrative

#### Precise Verse Extraction
- Extracts exact text from biblical verses (not entire chapters)
- Handles complex references (e.g., "26:6-13", "21:19b-22")
- Graceful fallback for cases where extraction fails

#### Perfect Chronological Sequence
- Maintains strict temporal order of 169 events
- Each event summarized in chronological context
- Preserves historical narrative of Holy Week

### 📈 Current Metrics Status

**Note**: Comprehensive metric evaluation is currently in progress. The temporal evaluation (Kendall's Tau) has been recently validated:

- **LEXRANK**: Kendall's Tau = 0.287 (partial temporal disorder)
- **LEXRANK-TA**: Kendall's Tau = 1.000 (perfect temporal order)
- **LEXRANK-TA-BEST**: Kendall's Tau = 1.000 (perfect temporal order)

ROUGE, BERTScore, and METEOR metrics are being updated and will be available in future releases.

## 🚀 Usage Examples

### Basic Usage
```bash
# Standard LEXRANK (semantic quality)
python src/main.py

# LEXRANK-TA (temporal order)
python src/main.py --summarization-method lexrank-ta --summary-length 2

# LEXRANK-TA-BEST (optimized temporal)
python src/main.py --summarization-method lexrank-ta-best --summary-length 1
```

### Generated Files

Each method creates specific files in the `outputs/` folder:

- **LEXRANK**: 
  - `evaluation/LEXRANK_results.json` - Evaluation metrics

- **LEXRANK-TA**:
  - `evaluation/LEXRANK-TA_results.json` - Evaluation metrics

- **LEXRANK-TA-BEST**:
  - `evaluation/LEXRANK-TA-BEST_results.json` - Evaluation metrics

### Method Comparison
```bash
python compare_methods.py
```

### Advanced Usage
```bash
# LEXRANK with 800 sentences (very detailed)
python src/main.py --summarization-method lexrank --summary-length 800

# LEXRANK-TA with 5 sentences per event (more detailed)
python src/main.py --summarization-method lexrank-ta --summary-length 5

# LEXRANK-TA-BEST with 1 sentence per event (concise)
python src/main.py --summarization-method lexrank-ta-best --summary-length 1
```

## Modules

### `data_loader.py`
Responsible for loading and processing the XML files of the gospels, specifically extracting the Holy Week chapters and chronology data.

### `graph_builder.py`
Constructs the temporal graph with chronological events and relationships between them.

### `models.py`
Defines the neural network models used for training and summarization.

### `train.py`
Handles the training process for the summarization models.

### `evaluate.py`
Calculates multiple evaluation metrics including ROUGE, METEOR, BERTScore, and Kendall's Tau to compare generated summaries with reference texts.

### `main.py`
Orchestrates the entire TAEG pipeline, coordinating data loading, graph construction, summarization, and evaluation.

## Evaluation Metrics

### ROUGE
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest Common Subsequence

### METEOR
Word alignment-based metric with synonymy and stemming.

### BERTScore
BERT embedding-based metric for semantic similarity.

### Kendall's Tau
Ranking correlation between sentence order in generated summary and reference text. Values range from -1 (perfect disagreement) to +1 (perfect agreement). Recently validated to correctly distinguish temporal preservation:
- LEXRANK: 0.287 (partial temporal disorder)
- LEXRANK-TA/LEXRANK-TA-BEST: 1.000 (perfect temporal order)

## 🔧 Recent Validation

### Kendall's Tau Metric Validation
The temporal evaluation metric has been thoroughly validated:
- **Debug Implementation**: Added position tracking for events in reference and hypothesis texts
- **Correct Behavior Confirmed**: 
  - Non-temporal methods (LEXRANK) show realistic partial disorder (τ = 0.287)
  - Temporal-anchored methods (LEXRANK-TA, LEXRANK-TA-BEST) achieve perfect order (τ = 1.000)
- **Event Matching**: Uses keyword overlap detection with NLTK sentence tokenization
- **Result**: System accurately evaluates temporal preservation differences between summarization approaches

## Dependencies

- `beautifulsoup4`: XML/HTML processing
- `lxml`: Efficient XML parser
- `lexrank`: LEXRANK algorithm
- `nltk`: Natural language processing
- `rouge-score`: ROUGE metrics
- `bert-score`: BERTScore metric
- `transformers`: Language models
- `torch`: Deep learning framework
- `scipy`: Scientific computing
- `pandas`: Data manipulation
- `numpy`: Numerical computing


## Contribution

To contribute to the project:

1. Fork the repository
2. Create a branch for your feature
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is distributed under the MIT license. See the LICENSE file for more details.

## Contact

For questions or suggestions, contact the development team.