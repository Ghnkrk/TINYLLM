# TINYLLM
TinyLLM is a compact, hands-on repository built to understand and train small-scale language models from the ground up. Instead of relying on high-level frameworks, it focuses on explicit model architecture, tokenizer training, dataset preparation, and step-controlled training loops.

The project experiments with training decoder-only language models in the 2M–20M parameter range on carefully chosen datasets, using modern practices like step-based early stopping, cosine learning rate schedules, and mixed-precision training. Each script is designed to be reusable and configurable, making it easy to compare different model sizes and datasets while keeping the training process transparent and reproducible.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Tokenizer Training**: Train a SentencePiece tokenizer for your dataset.
- **Dataset Processing**: Download and preprocess datasets for model training.
- **Configurable Architecture**: Easily customize model and training configurations.
- **Lightweight Setup**: A simple yet robust setup for experimentation.

---

## Installation

1. Clone the repository:

```bash
$ git clone https://github.com/Ghnkrk/TINYLLM.git
```

2. Install the required dependencies:

```bash
$ pip install -r requirements.txt
```

---

## Usage

### Dataset Processing

Run the following command to download and preprocess the dataset:

```bash
$ python data.py
```

### Tokenizer Training

To train the SentencePiece tokenizer, execute:

```bash
$ python tokenizer/tokenizer_train.py
```

### Model Training

Initiate model training:

```bash
$ python train.py --config config_{size}.json
```
Replace `{size}` with the desired configuration (e.g., `small` or `large`). You can customize the configuration files in the `configs/` directory.

### Parameter Check

To verify the model size before training, use:

```bash
$ python check_params.py
```

---

## Repository Structure

```
TINYLLM/
├── tokenizer/
│   └── tokenizer_train.py    # Script to train the tokenizer
├── configs/
│   └── config_small.json     # Configuration for small model
│   └── config_large.json     # Configuration for large model
├── data.py                   # Script to download and process dataset
├── train.py                  # Script to train the model
├── check_params.py           # Script to check model parameters
└── requirements.txt          # Dependency file
```

---

## Configuration

The repository includes two configuration files:

- **`config_small.json`**: Defines parameters for a smaller model.
- **`config_large.json`**: Defines parameters for a larger model.

You can edit these files to suit your needs. Pass the required configuration file to scripts, such as `train.py`, to modify training behavior.

---

## Architecture

This repository implements a moderately complex language model architecture. At its core, it includes:

- **Transformer Encoding Layers**: Stack of transformer layers for token processing.
- **Positional Encoding**: Encodes token positions to provide sequence awareness.
- **Attention Mechanisms**: Implements scaled dot-product attention for token interaction.
- **Feedforward Networks**: Fully connected layers for handling token embeddings.
- **Decoder**: Optional components tailored for downstream tasks.

---

## Screenshots

Add your screenshots below for better visualization. Replace these placeholders with your images:

- Training Workflow:

`![Training Screenshot](path/to/your/image.png)`

- Inference Results:

`![Inference Screenshot](path/to/your/image.png)`

---

## Contributing

Contributions are welcome! Feel free to fork the repository, make enhancements, and submit a pull request.

### Steps to Contribute

1. Fork the repository.
2. Create a new feature branch: `git checkout -b my-feature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin my-feature`.
5. Open a pull request.

---

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.

---
