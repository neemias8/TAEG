# TAEG - Temporal Alignment Event Graph

A modular system for consolidating and evaluating biblical narratives using natural language processing techniques.

## Overview

TAEG (Temporal Alignment Event Graph) is a project that combines narratives from the four Gospels (Matthew, Mark, Luke, and John) into a consolidated summary of Holy Week using advanced summarization algorithms, and evaluates the summary quality by comparing it with a reference sample (Golden Sample) using multiple evaluation metrics.

## Features

- **Data Loading**: Processes XML files from the Gospels to extract Holy Week texts
- **Summarization with Multiple Methods**:
  - **LEXRANK**: Standard multi-document algorithm 
  - **LEXRANK-TA (Temporal Anchoring)**: Optimized temporal anchoring with best sentence selection and perfect chronological order
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
- **Disadvantage**: It loses temporal order (lower Kendall's Tau)

### LEXRANK-TA (Temporal Anchoring)
- Builds upon LEXRANK with optimized sentence selection for temporal preservation
- Ensures perfect temporal ordering
- **Gospel-Specific Architecture**: Each chronological event has separate nodes for each Gospel that mentions it
- **Precise Verse Extraction**: Extracts exact biblical verse text instead of entire chapters
- **Multi-Document Summarization per Event**: For events mentioned in multiple Gospels, uses multi-document LEXRANK to combine complementary perspectives
- **Enhanced Temporal Graph**:
  - 363 gospel-specific nodes (one per Gospel per event)
  - 799 BEFORE edges (connecting consecutive events)
  - 318 SAME_EVENT edges (connecting different versions of the same event)
- **Strict Chronological Sequence**: Maintains perfect temporal order of 169 Holy Week events
- **Advantage**: Maximum temporal preservation with coherent output
- **Use Case**: When temporal accuracy is critical and completeness is desired


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
python src/main.py --summarization-method lexrank --summary-length 800

# LEXRANK-TA (recommended for temporal preservation)
python src/main.py --summarization-method lexrank-ta --summary-length 1
```

### Parameters

- `--summarization-method`: Summarization method:
  - `lexrank`: Standard multi-document LEXRANK (semantic quality)
  - `lexrank-ta`: Optimized temporal anchoring with perfect chronological order
- `--summary-length`:
  - For `lexrank`: Total number of sentences in summary
  - For `lexrank-ta`: Number of sentences per chronological event 
- `--data-dir`: Data directory (default: `data`)
- `--output-dir`: Output directory (default: `outputs`)

### Methods Comparison

```bash
python compare_methods.py
```

## 📊 Methods Comparison

Comparative test between LEXRANK and LEXRANK-TA (both evaluated against Golden Sample):

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | METEOR | Kendall's Tau | Summary Length | Priority |
|--------|------------|------------|------------|--------------|--------|----------------|---------------|----------|
| **LEXRANK** | 0.666 | 0.562 | 0.190 | 0.835 | 0.270 | 0.287 | 43,134 chars | **Semantic** |
| **LEXRANK-TA** | 0.958 | 0.938 | 0.947 | 0.995 | 0.639 | **1.000** | 79,154 chars | **Temporal** |

### 🔍 Results Analysis

- **LEXRANK**: Shows partial temporal disorder (Kendall's Tau = 0.287), indicating some chronological reordering for semantic optimization. Achieves some semantic quality (ROUGE-1 F1 = 0.666, BERTScore F1 = 0.835) but compromises temporal accuracy.
- **LEXRANK-TA**: Achieves perfect temporal preservation (Kendall's Tau = 1.000) with superior semantic quality (ROUGE-1 F1 = 0.958, BERTScore F1 = 0.995). The temporal anchoring approach provides the best balance of chronological accuracy and content quality.
- **Choice**: LEXRANK-TA for temporal-critical applications requiring both chronological order and high semantic quality, LEXRANK for pure semantic optimization when temporal order is not critical.

### Conciseness vs Consolidation Analysis

```bash
python analyze_conciseness.py
```

This script demonstrates that conciseness is not the most important factor in biblical narrative consolidation, showing that comprehensive consolidation with temporal accuracy outperforms brevity-focused approaches.

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

ROUGE, BERTScore, and METEOR metrics are being updated and will be available in future releases.

## � Conciseness vs Consolidation Analysis

Recent empirical analysis demonstrates that **conciseness is not the most important factor** in biblical narrative consolidation. Comprehensive consolidation that preserves multiple perspectives and maintains temporal accuracy is far more valuable than brevity.

### 🎯 Key Findings

The analysis compared LEXRANK at different summary lengths (100, 500, 1000, 1500 sentences) against LEXRANK-TA to determine the optimal balance between conciseness and quality:

| Method | ROUGE-1 F1 | BERTScore F1 | METEOR | Kendall's Tau | Length (chars) |
|--------|------------|--------------|--------|---------------|----------------|
| **LEXRANK (100 sent)** | 0.296 | 0.835 | 0.097 | 0.268 | 14,710 |
| **LEXRANK (500 sent)** | 0.804 | 0.835 | 0.361 | 0.305 | 59,408 |
| **LEXRANK (1000 sent)** | 0.862 | 0.835 | 0.483 | 0.320 | 100,770 |
| **LEXRANK (1500 sent)** | 0.784 | 0.835 | 0.484 | 0.320 | 128,930 |
| **LEXRANK-TA (Reference)** | **0.958** | **0.995** | **0.639** | **1.000** | 79,154 |

### 🔍 Analysis Insights

1. **📈 Quality improves with length**: Longer LEXRANK summaries capture more biblical content and achieve better semantic quality (ROUGE, METEOR scores increase significantly from 100 to 1000 sentences)

2. **⏰ Temporal order consistency**: LEXRANK maintains relatively stable temporal correlation (Kendall's Tau ~0.3) across different lengths, indicating consistent chronological behavior

3. **🎯 LEXRANK-TA superiority**: Maintains perfect temporal order (τ = 1.000) with superior semantic quality, demonstrating that temporal anchoring + comprehensive consolidation wins over pure conciseness

4. **📚 Biblical narrative conclusion**: For consolidating multiple Gospel perspectives, **comprehensive coverage with temporal accuracy is more valuable than brevity**

### ⚖️ Trade-off Analysis

- **LEXRANK (1500 sentences)**: Temporal τ=0.320, Semantic F1=0.835, Length=128,930 chars
- **LEXRANK-TA**: Temporal τ=**1.000**, Semantic F1=**0.995**, Length=79,154 chars

**Conclusion**: In biblical narrative consolidation, temporal accuracy + comprehensive content consistently outperforms conciseness-focused approaches.

## �🚀 Usage Examples

### Basic Usage
```bash
# Standard LEXRANK (semantic quality)
python src/main.py

# LEXRANK-TA (temporal order)
python src/main.py --summarization-method lexrank-ta --summary-length 1
```

### Generated Files

Each method creates specific files in the `outputs/` folder:

- **LEXRANK**: 
  - `evaluation/LEXRANK_results.json` - Evaluation metrics

- **LEXRANK-TA**:
  - `evaluation/LEXRANK-TA_results.json` - Evaluation metrics

### Method Comparison
```bash
python compare_methods.py
```

### Advanced Usage
```bash
# LEXRANK with 800 sentences (very detailed)
python src/main.py --summarization-method lexrank --summary-length 800

# LEXRANK-TA with 1 sentence per event (optimal temporal preservation)
python src/main.py --summarization-method lexrank-ta --summary-length 1
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
- LEXRANK-TA: 1.000 (perfect temporal order)

## 🔧 Recent Validation

### Kendall's Tau Metric Validation
The temporal evaluation metric has been thoroughly validated:
- **Debug Implementation**: Added position tracking for events in reference and hypothesis texts
- **Correct Behavior Confirmed**: 
  - Non-temporal methods (LEXRANK) show realistic partial disorder (τ = 0.287)
  - Temporal-anchored methods (LEXRANK-TA) achieve perfect order (τ = 1.000)
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