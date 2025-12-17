# Radar Simulation ML

Machine learning models for predicting aircraft trajectories from radar data.
For my bachelor thesis.

## Project Structure

```
├── data/                   # Training data (.pt files)
├── models/                 # Saved model weights
├── src/
│   ├── tools/              # Utilities (codecs, data processing)
│   ├── training/           # Model training scripts
│   └── evaluation/         # Model evaluation scripts
└── pyproject.toml
```

## Setup

```bash
# Install dependencies using uv
uv sync
```

## Usage

### 1. Build Training Data

Extract trajectories from SQLite database:

```bash
uv run src/tools/build_training_set.py
```

### 2. Train Models

**LSTM model** (sequence-to-sequence):

```bash
uv run src/training/train_lstm.py
```

**Simple NN model** (single-step prediction):

```bash
uv run src/training/train_nn.py
```

### 3. Validate Models

Test LSTM model:

```bash
uv run src/validation/test_lstm.py
```

Test NN models:

```bash
uv run python src/validation/test_naive.py
uv run python src/validation/test_normalized.py
```

## Data Format

Each flight point consists of 5 features (z-score normalized):

-   `x`, `y` - Cartesian position (from polar rho/theta)
-   `vx`, `vy` - Velocity components (from speed/heading)
-   `fl` - Flight level

## Models

-   **NextStepLSTM**: Recurrent model for trajectory prediction
-   **MovementPredictor**: Feedforward NN for single-step prediction
