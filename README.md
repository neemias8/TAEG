# TAEG (Temporal Alignment Event Graph)

TAEG implements a graph-based approach for abstractive multi-document summarisation of overlapping Gospel narratives. The project aligns passages from the four Gospels to a shared Holy Week chronology, builds a temporal event graph, and trains neural models to produce coherent summaries.

## Objective

Develop the TAEG pipeline that:
- parses the provided XML sources into an aligned event dataset;
- constructs a temporal event graph with explicit cross-gospel edges;
- encodes the graph with Graph Attention Networks and Transformer-based language models;
- benchmarks against strong sequence baselines (PEGASUS, PRIMERA) and an extractive LexRank variant;
- evaluates generated summaries with ROUGE, BERTScore, and temporal coherence diagnostics.

## Data Requirements

Place the following XML resources inside `data/`:

- `ChronologyOfTheFourGospels_PW.xml` – master chronology with 144 Holy Week events.
- `EnglishNIVMatthew40_PW.xml` – Passion Week by the Gospel of Matthew (NIV translation).
- `EnglishNIVMark41_PW.xml` – Passion Week by the Gospel of Mark (NIV translation).
- `EnglishNIVLuke42_PW.xml` – Passion Week by the Gospel of Luke (NIV translation).
- `EnglishNIVJohn43_PW.xml` – Passion Week by the Gospel of John (NIV translation).

The chronology file lists verse references for each Gospel; the loader extracts and aligns the underlying verses when all five XML files are available.

## Project Layout

```
TAEG/
├── data/                        # XML sources (ignored by Git)
├── notebooks/
│   └── 01_data_exploration.ipynb # Exploratory data notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # XML parsing and verse alignment
│   ├── graph_builder.py          # Temporal event graph construction
│   ├── models.py                 # TAEG, PEGASUS, PRIMERA, LexRank definitions
│   ├── train.py                  # Training loop for the TAEG model
│   ├── evaluate.py               # Metrics, analysis, and plotting
│   └── main.py                   # Orchestrates data → graph → train → evaluate
├── requirements.txt              # Python dependencies
├── setup.py / pyproject.toml     # Package metadata
└── README.md
```

## Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd TAEG
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate
   ```

3. Install dependencies and local package:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

## Usage

### Full Pipeline
Run the end-to-end process (data validation, graph construction, model training, evaluation, and reporting):
```bash
python src/main.py --mode complete --config config/default.yaml
```

### Train Only
```bash
python src/main.py --mode train --epochs 50 --batch_size 8
```

### Evaluate Existing Models
```bash
python src/main.py --mode evaluate --model_path outputs/checkpoints/best_model.pt
```

### Ablation Study
```bash
python src/main.py --mode ablation
```

### Data Exploration Notebook
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Experiments and Baselines

- **TAEG (proposed)** – Graph encoder (GAT) + Transformer decoder; initial node features from a Hugging Face text encoder.
- **PEGASUS** – Abstractive baseline using concatenated Gospel texts.
- **PRIMERA** – Multi-document abstractive baseline.
- **LexRank** – Extractive graph-based baseline.

Ablations include removing temporal edges and disabling the graph encoder to quantify structural contributions.

## Evaluation Metrics

- **ROUGE-1/2/L** via `rouge_score`.
- **BERTScore** via `bert-score`.
- **Temporal coherence** diagnostics implemented in `evaluate.py`.

Evaluation artefacts (metrics, plots, comparison tables) are written to `outputs/evaluation/`.

## Configuration

The YAML configuration supplied via `--config` overrides defaults in `main.py`. Key sections include data paths, model hyperparameters, training schedule, and evaluation toggles.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-change`.
3. Commit with descriptive messages and add tests if feasible.
4. Submit a pull request for review.

## License

This project is released under the MIT License. Refer to `LICENSE` for details.

## Contact

- Author: Your Name
- Email: your.email@example.com
- Project page: <https://github.com/your-user/TAEG>
