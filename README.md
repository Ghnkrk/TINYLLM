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

**Tokenizer Training:** Train a SentencePiece tokenizer directly on the target dataset to ensure the vocabulary matches the data the model is trained on.

**Dataset Processing:** Load, filter, and chunk raw text datasets into fixed-length sequences suitable for autoregressive language modeling.

**Configurable Architecture:** Define model size, depth, and training behavior through configuration files, making it easy to experiment with different setups.

**Lightweight Setup:** Keep the codebase minimal and explicit, focusing on clarity and control rather than heavy abstractions.

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

### Tokenizer Training

To train the SentencePiece tokenizer, execute:

```bash
#Edit the config file path in the code accordingly
$ python tokenizer/tokenizer_train.py
```

### Model Training

Initiate model training:

```bash
#Configure the config file as per your requirement.
$ python train.py --config config.json
```

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
│   └── config_2M.json     # Configuration for small model
│   └── config_15M.json     # Configuration for large model
├── data.py                   # Script to download and process dataset
├── train.py                  # Script to train the model
├── check_params.py           # Script to check model parameters
└── requirements.txt          # Dependency file
```

---

## Configuration

The repository includes two configuration files:

- **`config_2M.json`**: Defines parameters for a smaller model.
- **`config_15M.json`**: Defines parameters for a larger model.

You can edit these files to suit your needs. Pass the required configuration file to scripts, such as `train.py`, to modify training behavior.

---

## Architecture

This repository implements a moderately complex language model architecture. At its core, it includes:

-**Transformer Layers**: A stack of transformer blocks composed of self-attention and feedforward sublayers for contextual token processing.

-**Rotary Positional Encoding (RoPE)**: Applied within the attention mechanism to inject relative positional information without explicit position embeddings.

-**Self-Attention Mechanism**: Uses scaled dot-product self-attention to allow each token to attend to other tokens in the sequence.

-**Feedforward Networks**: Position-wise feedforward layers that transform token representations after attention, enabling non-linear feature learning.

-**Decoder-Only Architecture**: Uses masked self-attention for autoregressive language modeling, where each token attends only to past tokens (no encoder–decoder cross-attention).

---

## Screenshots

Add your screenshots below for better visualization. Replace these placeholders with your images:

- Autoregressive Architecture:

<img width="847" height="1080" alt="Capture" src="https://github.com/user-attachments/assets/10c9b0ce-0027-4345-8e5f-7a419d929b86" />


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
