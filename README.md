# FedDoG

FedDoG for one-shot federated learning. Supports DomainNet, PACS and OfficeHome, with Stable Diffusion 1.5 and SDXL. FedDoG includes domain-level and class-level representations (`--granularity`).

## Step 1: Install the environment

Install Miniconda or Anaconda first.

Run the following commands from the project root. `-n feddog` sets the environment name to `feddog`, overriding the name in the YAML file:

```bash
conda env create -n feddog -f environment.yml
conda activate feddog
```

Model weights are downloaded from Hugging Face on first use.

## Step 2: Prepare the datasets

All three datasets can be prepared using the download utility:

```bash
python datasets/download_datasets.py --datasets DomainNet PACS OfficeHome

# Download DomainNet only.
python datasets/download_datasets.py --datasets DomainNet
```

## Step 3: Run experiments

Run commands from the project root; see [`run.sh`](run.sh) for examples grouped by dataset.

```bash
# FedDoG: extract client representations, generate images, train classifier.
python main.py --algorithm feddog --stage all --dataset DomainNet --gamma 2 --device cuda:0

# Class-level FedDoG.
python main.py --algorithm feddog --granularity class --stage all --dataset DomainNet --gamma 1 --device cuda:0

# PromptOnly: generate images using class prompts and train classifier.
python main.py --algorithm promptonly --stage all --dataset PACS --device cuda:0
```

### Stages

Use `--stage extract`, `generate`, `train`, or `all` (PromptOnly has no `extract` stage). Keep dataset, seed, client count, granularity and model options consistent across stages.

To train on existing generated images, use `--stage train --synthetic-data-dir <run>`; FedDoG also requires the matching representation payload.

### Key options

| Option | Usage |
| --- | --- |
| `--sd-version sdxl` | Use SDXL instead of SD1.5. |
| `--generation-batch-size` | Candidates per diffusion call (default: `4`); use `1` if GPU memory is limited. |
| `--shots-per-class 8` | Match the OfficeHome experiment setup. |

See `python main.py --help` for all options.

### Outputs

- Generated images: `generated_data/<dataset>/`
- Client representations: `outputs/models/`
- Accuracy metrics: `outputs/results/`

Classifier weights are saved only with `--save-global-classifier`.

## Source layout

- `main.py`: command-line arguments and algorithm selection.
- `run.sh`: commented experiment commands grouped by dataset.
- `algorithms/feddog.py`: FedDoG representation extraction and image generation.
- `algorithms/promptonly.py`: PromptOnly generation baseline.
- `algorithms/base.py`: shared classifier training and accuracy calculation.
- `datasets/`: dataset definitions, loaders, client partitions and download utility.
