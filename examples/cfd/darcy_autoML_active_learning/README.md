# Scalable PDE Surrogate Modeling: Darcy Flow with FNO/AFNO + AutoML & Active Learning

## Prerequisites
For environment or Docker instructions, please consult **[GETTING_STARTED.md](./GETTING_STARTED.md)**.
After completing those steps, you can launch the Modulus container and run the notebooks by:
```bash
# If container isn't running, start it
docker start my_modulus_container
# Attach to it
docker exec -it my_modulus_container bash
# Move into the example folder
cd examples/cfd/darcy_autoML_active_learning
# Launch Jupyter (published on port 8888)
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```
Then open your browser at http://localhost:8888 (with the token link printed in the console), navigate to notebooks/darcy_autoML.ipynb or darcy_active_learning.ipynb.

## Introduction
This repository demonstrates a **multi-faceted Physics-AI pipeline** for **Darcy Flow** PDE surrogate modeling. It features:
1. **A full data→model pipeline** for Darcy Flow surrogates (FNO/AFNO).  
2. **AutoML** hyperparameter tuning (via Optuna or similar).  
3. **Offline Active Learning** using MC-Dropout to identify high-uncertainty PDE samples.  

While Darcy Flow serves as our **example PDE**, the underlying **architecture** (modular \`src/\` code, notebooks, MLFlow integration) is designed to scale to **broader PDE problems** in engineering or scientific HPC workflows.  

## Notebooks Overview
1. **Notebook 1:** [darcy_autoML.ipynb](./notebooks/darcy_autoML.ipynb)  
   - **Introduction & Vision**: Explains the rationale for PDE-based surrogate modeling, plus how we can unify Darcy Flow, neural operators (FNO/AFNO), and an AutoML approach.  
   - **Data Generation & Loading**: Either synthetically produce Darcy Flow fields or load them from `.pt` files.  
   - **Surrogate Model Definition**: Demonstrates constructing a **FNOWithDropout** or an AFNO operator from the `src/models/` folder.  
   - **Hyperparameter Tuning (AutoML)**: Shows how to systematically search over PDE operator hyperparameters (modes, width, depth, etc.) for optimal results.  
   - **Training Execution**: A configurable training loop (`src/ModelTrainingLoop.py`) logs metrics (optionally to MLFlow), and can handle HPC or local usage.  
   - **Performance Visualization**: Minimal or extended visualization (train/val losses, PDE predictions).

2. **Notebook 2:** [darcy_active_learning.ipynb](./notebooks/darcy_active_learning.ipynb)  
   - **Offline Active Learning**: Builds on the **trained PDE surrogate** from Notebook 1.  
   - **MC-Dropout** for Uncertainty: Multiple forward passes yield mean & variance for each PDE input.  
   - **Selecting Top-K**: Identifies which PDE fields are most “uncertain,” potentially requiring additional HPC solver runs or partial retraining.  
   - **Saving**: Optionally store the top-K uncertain samples in `.pt` format for further data augmentation.  

> **Note**: If you’d like more details on environment setup, Docker usage, or how to run these notebooks in a local vs. HPC scenario, see **[GETTING_STARTED.md](./GETTING_STARTED.md)**.  

## Repository Structure

```
darcy_autoML_active_learning/
├─ notebooks/
│   ├─ darcy_autoML.ipynb           # Notebook 1: Surrogate + AutoML
│   ├─ darcy_active_learning.ipynb  # Notebook 2: Offline AL with MC-Dropout
├─ src/
│   ├─ darcy_automl_active_learning/
│   │   └─ data_loading.py
│   ├─ models/
│   │   └─ fno_with_dropout.py
│   │   ├─ model_factory.py
│   │   ├─ ModelTrainingLoop.py
│   ├─ automl
│   ├─ automl.py
│   ├─ AL/
│   │   ├─ mc_dropout_estimator.py
│   │   └─ offline_al_demo.py (optional)
│   └─ visualization.py
├─ config/
│   └─ config.yaml
├─ ...
└─ GETTING_STARTED.md
└─ requirements.txt
└─ pyproject.toml
```

### Key Highlights
- **Data & Surrogate**: Illustrates PDE data ingestion (e.g., uniform grids, Darcy2D) and operator-based networks (FNO, AFNO).  
- **AutoML**: Uses a flexible search approach to optimize hyperparameters (learning rate, modes, etc.), easily extended to HPC or multi-GPU usage.  
- **Active Learning**: Demonstrates a straightforward **offline** approach with MC-Dropout to rank PDE samples by uncertainty.  
- **MLFlow**: Optionally logs training & tuning metrics. You can run `mlflow ui -p 2458` to visualize them in a local browser, or set up SSH port-forwarding on HPC.

### Using MLFlow (Optional)
If you enable MLFlow logging (e.g., in `config.yaml` or directly in the notebook cells), you can **monitor training** or AL runs in real time:

1. **Local**:  
   ```bash
   mlflow ui -p 2458
   ```
   Then open [http://127.0.0.1:2458](http://127.0.0.1:2458).  
2. **Remote HPC**:  
   - SSH with `-L 8080:127.0.0.1:8080`  
   - On remote: `mlflow ui --host 0.0.0.0 --port 8080`  
   - Local browser → `localhost:8080`

See the examples in both notebooks for how to integrate MLFlow calls.

### Next Steps & Vision
Both notebooks emphasize **scalability**:  
- Additional PDEs (e.g., subgrid turbulence, multi-physics PDEs) can use the same pipeline with minor changes in data loading & operator choice.  
- HPC synergy is achievable via distributed data generation or multi-GPU neural operator training.  
- The entire approach can be integrated into a larger **Physics-AI** solution, combining an **ontology-based** data engine, advanced model selection, and real-time HPC embeddings.
