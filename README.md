
# **TAEG: From Summarization to Consolidation with Temporal Alignment Event Graphs**


This repository contains the official implementation and resources for the paper **"From Summarization to Consolidation: A Temporal Alignment Event Graph (TAEG) Approach for Unifying Narratives"**, currently under review for the LREC-COLING 2026 conference.

## **Abstract**

In Multi-Document Summarization (MDS) of long narratives, such as legal testimonies or historical accounts, the primary goal is often not compression but coherent **consolidation**.1 Standard graph-based methods like LexRank, focused on semantic similarity, fail to preserve chronological flow. This paper reframes the task as narrative consolidation and introduces the

**Temporal Alignment Event Graph (TAEG)**, a novel structure that explicitly models chronology and event alignment. By applying the LexRank algorithm to a TAEG, our method serves as a version selection mechanism, choosing the most central representation of each event in its correct chronological position. This approach guarantees perfect temporal ordering (Kendall's Tau of 1.000) by design. In a study on the four Biblical Gospels, our TAEG-based method dramatically improves content metrics (e.g., \+357.2% in ROUGE-L), demonstrating that for complex narratives, comprehensive unification is a more meaningful goal than simple summarization.1

## **The Core Problem: Summarization vs. Narrative Consolidation**

The central premise of this work is a fundamental reframing of how we process multiple, overlapping narrative documents. The goal is not to make the story shorter, but to make it **whole**.

Traditional Multi-Document Summarization (MDS) is defined by its focus on **conciseness**.1 However, in contexts like a criminal investigation with multiple witness testimonies or a historical analysis of overlapping accounts like the Biblical Gospels, the primary objective is to produce a single, unified, and chronologically sound narrative. The final text must eliminate redundancy while integrating crucial details from all sources into a cohesive whole.1

Classic graph-based algorithms like LexRank are fundamentally mismatched for this task. By optimizing for semantic centrality, they inherently ignore the chronological flow of the narrative, resulting in a collection of salient but temporally disordered facts.1

This project advocates for a paradigm shift from summarization to **narrative consolidation**, where coherence, completeness, and temporal integrity are prioritized over brevity.

## **The Solution: Temporal Alignment Event Graph (TAEG)**

To achieve true narrative consolidation, we introduce the **Temporal Alignment Event Graph (TAEG)**, a structure that prioritizes temporal order and event alignment over simple semantic similarity.

Unlike standard methods that infer structure from textual similarity, the TAEG's construction is driven by external knowledge—a pre-defined, canonical chronology of events that serves as a structural backbone.1

### **TAEG Architecture**

The TAEG is a multi-relational graph designed to separate the challenges of chronological ordering and version selection:

* **Nodes**: A distinct node is created for each *version* of a canonical event. For example, if an event is described in Matthew, Mark, and Luke, three separate nodes are created for that single event.1  
* **Edges**: The graph contains two functionally distinct types of edges:  
  1. **Temporal Edges (BEFORE)**: *Directed* edges that connect nodes representing sequential events *within the same source document*. These edges form the known chronological backbone of each narrative.  
  2. **Anchoral Edges (SAME\_EVENT)**: *Undirected* edges that interconnect all nodes (versions) that refer to the *same canonical event*, creating a cluster for each event in the timeline.1

This dual-edge architecture decouples the two primary challenges: BEFORE edges solve the sequencing problem, while SAME\_EVENT edges isolate the version selection problem.

## **Methodology & Results**

We used the exact same centrality algorithm, LexRank, for both the standard baseline and our TAEG-based system. The only variable was the underlying graph structure. When applied to the TAEG, LexRank is repurposed into a powerful **version selection engine**.1

The results are definitive. The TAEG-based approach is not just incrementally better; it represents a categorical improvement, guaranteeing perfect temporal coherence *by design*.

| Metric | Baseline (Standard LexRank) | TAEG-LexRank | Improvement |
| :---- | :---- | :---- | :---- |
| ROUGE-1 F1 | 0.887 | 0.958 | \+8.0% |
| ROUGE-2 F1 | 0.712 | 0.938 | \+31.7% |
| ROUGE-L F1 | 0.207 | 0.947 | **\+357.2%** |
| BERTScore F1 | 0.835 | 0.995 | \+19.1% |
| METEOR | 0.453 | 0.639 | \+41.0% |
| Kendall's Tau | 0.320 | **1.000** | \+212.5% |

The perfect Kendall's Tau score is an architectural property of the TAEG, not a learned outcome. This perfect ordering is the direct cause of the massive **\+357.2% improvement in ROUGE-L**, which measures the longest common subsequence.1

### **Conciseness vs. Consolidation Analysis**

The results for the standard LexRank baseline in the table above reflect a parameter setting of 750 sentences. To prove that the TAEG's superiority is structural and not a matter of parameter tuning, the table below shows the baseline's performance across various summary lengths.

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | METEOR | Kendall's Tau | Length (chars) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **LexRank Baseline** |  |  |  |  |  |  |  |
| *100 sentences* | 0.296 | 0.263 | 0.129 | 0.835 | 0.097 | 0.268 | 14,710 |
| *500 sentences* | 0.804 | 0.655 | 0.206 | 0.835 | 0.361 | 0.305 | 59,408 |
| *1000 sentences* | 0.862 | 0.728 | 0.199 | 0.835 | 0.483 | 0.320 | 100,770 |
| *1500 sentences* | 0.784 | 0.733 | 0.188 | 0.835 | 0.484 | 0.320 | 128,930 |
| **TAEG-LexRank** | **0.958** | **0.938** | **0.947** | **0.995** | **0.639** | **1.000** | **79,154** |

This analysis clearly demonstrates that simply increasing the number of sentences does not address the fundamental problem of narrative coherence. While some metrics improve up to a point, the temporal coherence (Kendall's Tau) remains consistently low. Even at its peak performance, the standard LexRank approach fails to come close to the quality and temporal integrity of the TAEG-based method. This reinforces our central argument: for long and complex narratives, **comprehensive coverage and chronological soundness are far more critical than mere conciseness**.1

## **The Gospel Consolidation Language Resource**

To facilitate this research, we have developed and publicly released the **Gospel Consolidation Language Resource**. The dataset comprises the English New International Version (NIV, 2011\) texts of the four Gospels, mapped to 169 canonical events from the Holy Week, and a high-quality, manually created reference consolidation (the "Golden Sample").1

### **Language and Version Agnostic Format**

A crucial design choice was the use of a 'book:chapter:verse' system for alignment. This decouples the chronological structure from any specific translation or language. This means other researchers can easily apply our framework to the Gospels in different languages or biblical versions (e.g., KJV, ESV) without recreating the temporal alignment from scratch.1

The system is designed to parse any XML file where book, chapter, and verse identifiers are clearly tagged as attributes. As long as the verse references can be parsed, the TAEG can align them regardless of the specific XML schema or textual content.

For example, consider Matthew 21:1 from two different versions in a simple XML format:

**NIV (New International Version) XML:**

XML

\<bible version\="NIV"\>  
  \<book name\="Matthew"\>  
    \<chapter number\="21"\>  
      \<verse number\="1"\>As they approached Jerusalem and came to Bethphage on the Mount of Olives, Jesus sent two disciples,\</verse\>  
    \</chapter\>  
  \</book\>  
\</bible\>

**KJV (King James Version) XML:**

XML

\<bible version\="KJV"\>  
  \<book name\="Matthew"\>  
    \<chapter number\="21"\>  
      \<verse number\="1"\>And when they drew nigh unto Jerusalem, and were come to Bethphage, unto the mount of Olives, then sent Jesus two disciples,\</verse\>  
    \</chapter\>  
  \</book\>  
\</bible\>

Our system aligns both passages to the same canonical event by parsing the attributes book name="Matthew", chapter number="21", and verse number="1", making the framework highly adaptable and reusable across different XML-formatted biblical texts.

## **Getting Started**

### **Installation**

To set up the environment and install the required dependencies, follow these steps:

Bash

\# Clone the repository  
git clone https://github.com/neemias8/TAEG.git  
cd TAEG

\# Install dependencies  
pip install \-r requirements.txt

### **Usage**

To run the narrative consolidation process and replicate the paper's core findings, use the following command:

Bash

\# Run the consolidation using the TAEG-LexRank method  
python main.py \--method taeg \--output consolidated\_narrative.txt

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
- LEXRANK: 0.320 (partial temporal disorder)
- LEXRANK-TA: 1.000 (perfect temporal order)

## 🔧 Recent Validation

### Kendall's Tau Metric Validation
The temporal evaluation metric has been thoroughly validated:
- **Debug Implementation**: Added position tracking for events in reference and hypothesis texts
- **Correct Behavior Confirmed**: 
  - Non-temporal methods (LEXRANK) show realistic partial disorder (τ = 0.320)
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