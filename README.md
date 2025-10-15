
# Project Title

This repository accompanies the MS Thesis submission at Liverpool John Moores University (LJMU), 2025.

AI-Generated Sustainable Supply Chain Models.
This repository implements the research and experiments described in the MS Thesis
“AI-Generated Sustainable Supply Chain Models Using Variational Autoencoders and Agentic AI”

The project develops a hybrid Variational Autoencoder (VAE) and AI-Agent-based Route Generator to design multi-segment logistics routes optimized for low-carbon emissions while maintaining feasibility.



## Abstract

This project presents a hybrid AI framework for generating sustainable, multi-segment logistics routes that minimize carbon emissions while maintaining practical feasibility. A Variational Autoencoder (VAE) was developed to predict optimal distance–emission trade-offs across transportation modes. These predictions were then integrated with a GPT-4.0-based AI agent, which generated realistic multi-modal routes (e.g., combining Cargo Train and Container Ship) under emission and distance constraints.

The combined approach demonstrated that generative models and large language models (LLMs) can jointly improve logistics decision-making by balancing sustainability and operational practicality. Experimental results showed an average emission reduction of 53% across valid generated routes, confirming the potential of AI-driven optimization for green supply chains.

The findings demonstrate that AI-driven multi-modal logistics design can significantly reduce emissions, supporting the Paris net-zero goals.
## Technology Stack

| Category                          | Tools / Libraries            | Purpose                                                           |
| --------------------------------- | ---------------------------- | ----------------------------------------------------------------- |
| **Programming Language**          | Python 3.9                   | Core development and experimentation                              |
| **Deep Learning Framework**       | PyTorch & HuggingFace                     | Training the Variational Autoencoder (VAE)                        |
| **Machine Learning Utilities**    | scikit-learn, Joblib         | Feature scaling, encoding, and model persistence                  |
| **Data Processing**               | Pandas, NumPy, OpenPyXL      | Data manipulation, preprocessing, and analysis                    |
| **Visualization**                 | Matplotlib, Seaborn          | Data analysis, result visualization, and thesis figures           |
| **AI Agent / LLM**                | OpenAI GPT-4.0           | Used for intelligent multi-segment route generation and reasoning |
| **Experiment Management**         | Jupyter Notebook, TQDM       | Interactive workflow and training visualization                   |
| **Environment & Reproducibility** | Virtualenv, requirements.txt | Environment isolation and reproducibility                         |

## Environment Setup

To ensure full reproducibility of the experiments and results, all dependencies used across the notebooks are listed in the accompanying requirements.txt file.

### Create a virtual environment

  python -m venv .venv
- source .venv/bin/activate    # (on macOS / Linux)
- .venv\Scripts\activate       # (on Windows)

### Install dependencies

pip install -r requirements.txt

This installs all core libraries required for:

Model training (PyTorch, scikit-learn)

Data handling (Pandas, NumPy, OpenPyXL)

Visualization (Matplotlib, Seaborn)

Utilities (Joblib, TQDM)

The same environment was used during all thesis experiments and figure generation to ensure consistent reproducibility across notebooks.
## Repository Structure

| File                                    | Purpose                                                                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **`data_preprocessing.ipynb`**          | Cleans and prepares raw supply chain datasets, handles missing values, and standardizes textual attributes.                    |
| **`data_transform_feature_engg.ipynb`** | Performs feature engineering and scaling (e.g., distance normalization, emission intensity).                                   |
| **`eda.ipynb`**                         | Exploratory Data Analysis – visualizes transport mode distribution, distance-emission relationships, and correlation heatmaps. |
| **`vae_model.ipynb`**                   | Defines, trains, and evaluates the Variational Autoencoder used for predicting optimal distance and emission values.           |
| **`ai_agent_route_generation.ipynb`**   | Generates multi-segment routes using AI-Agent logic, conditioned on the VAE’s predicted parameters.                            |
| **`sample_route_generator_demo.ipynb`** | Demonstrates individual route generation with real inputs and VAE predictions.                                                 |
| **`inference_plots.ipynb`**             | Produces final evaluation plots – emission reduction, mode-shift visualization, and route validity summaries.                  |

## Data and Models

```plaintext
data/
├── raw/                               → Original datasets
├── cleaned/                           → Cleaned and scaled data
│   ├── scaler.pkl                     → Fitted StandardScaler
│   ├── source_label_encoder.pkl       →   Label encoder for source locations
│   ├── destination_label_encoder.pkl  →   Label encoder for destination locations
│   └── transportation_mode_label_encoder.pkl  →   Label encoder for transport modes

plots/
└──      → Exploratory Data Analysis (EDA) plots and figures

models/
└── best_model.pt                    → Trained Variational Autoencoder (VAE) weights

results/
├── vae_outputs/                     → Predicted emission, distance, and mode outputs
├── generated_routes/                → Generated multi-segment routes and validation results
└── inference_plots/                 → Final inference plots and visualizations used in the thesis

```




## Execution Order

To reproduce the workflow:

1️⃣ Preprocessing

Run → data_preprocessing.ipynb

2️⃣ Feature Engineering

Run → data_transform_feature_engg.ipynb

3️⃣ Exploratory Data Analysis

Run → eda.ipynb

4️⃣ VAE Training & Inference

Run → vae_model.ipynb

5️⃣ AI Agent Route Generation

Run → ai_agent_route_generation.ipynb

6️⃣ Demonstration & Visualization

Run → sample_route_generator_demo.ipynb
Run → inference_plots.ipynb

Each notebook can be executed independently but follows the above order for full reproducibility.
## Methodology Overview

1. ### Variational Autoencoder (VAE)

Input: Source, destination, transport mode, distance, emission.

Output: Predicted low-emission distance and transport class (mode).

Trained using PyTorch, AdamW optimizer, and ReduceLROnPlateau scheduler.

Fine-tuned for low-rank adaptation (LoRA) for efficient learning.

2. ### AI-Agent Route Generation

Uses predicted parameters from the VAE.

Constructs multi-segment routes (air, sea, rail, or road) ensuring total emission and distance ≤ predicted thresholds.

Evaluates route feasibility using validity rules and emission continuity checks.

3. ### Result Visualization

Comparison between real vs generated emissions.

Mode-shift analysis (e.g., Air Freight → Cargo Train).

Route validity statistics and emission reduction percentages.
## Citation

If you reference this project, please cite:

Kadangara, S. (2025).
AI-Generated Sustainable Supply Chain Models using Variational Autoencoders and Agentic AI.
MS Thesis, Liverpool John Moores University.
## Notes

- All core logic is implemented in Jupyter Notebooks for transparency and ease of understanding.

- Each notebook can be executed sequentially to reproduce the results shown in the thesis.

- Model files (best_model.pt, encoders, scaler) and output figures are provided for reproducibility.

- The environment can be replicated using the included requirements.txt.
## Example Results


| Metric | Description | Value |
|--------|--------------|--------|
| **Average Emission Reduction** | Compared to baseline routes | ~53% |
| **Valid Routes - Rank-1** | Feasible routes under VAE constraints | 85.7% |
| **Common Mode Shift** | Air Freight → Cargo Train | 70% |

> Detailed experimental results and visualizations are provided in Chapter 5 of the thesis and in the `results/inference_plots` folder.

## Sample Outputs

| Folder                      | Description                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `results/vae_outputs/`      | Contains VAE predictions for emission–distance relationships.                      |
| `results/generated_routes/` | Includes generated routes, emission comparisons, and validity evaluation.          |
| `results/inference_plots/`            | Contains all thesis-ready visualizations used in Chapter 5 (Results & Discussion). |

## Author and Supervisor

| Role           | Name                 |
| -------------- | -------------------- |
| **Author**     | **Sajana Kadangara** |
| **Research Supervisor** | **Akhil Kumar**      |

## License

This repository is provided for academic and research use only.
