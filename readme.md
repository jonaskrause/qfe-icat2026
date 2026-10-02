# Quantum Functional Evolution — ICAT 2026

Official code and experimental results for the paper:

> **Quantum Functional Evolution: Information-Shaped Unitary Dynamics for Variational Quantum State Preparation**  
> Jonas Krause and Rodrigo Pasti  
> 8th International Conference on Applied Technologies (ICAT 2026)

Quantum Functional Evolution (QFE) is an information-shaped evolutionary framework for variational quantum state preparation. The method combines population-based optimization, Functional Information derived from state fidelity, parameter-shift-inspired local refinement, angular recombination, mutation, and late-stage population contraction.

The experiments evaluate GHZ-state preparation across 120 benchmark configurations covering 3–10 qubits, three circuit depths, and five random seeds.

## Repository contents

```text
.
├── README.md
├── qfe_icat_2026.py       # QFE implementation and comparison methods
├── run_icat_2026_grid.sh        # Script for executing the experimental grid
└── icat_2026_results-10qb.zip   # Complete logs, metrics, plots, and configurations
```

## Methods

The implementation contains the following modes:

| Mode | Paper label | Description |
|---|---:|---|
| `FIDELITY_ONLY` | EFO | Evolutionary optimization using fidelity as the sole score |
| `QFE_SCORE` | FIS | Functional Information-shaped selection |
| `FLOW_ONLY` | CEE | FI-shaped selection and population contraction without local refinement |
| `GRADIENT_ONLY` | VGR | Parameter-shift refinement of fidelity without FI shaping or contraction |
| `QFE_UNITARY_FLOW` | QFE | Complete proposed method |
| `RANDOM` | — | Random-search diagnostic control; not included in the five-method aggregate comparison |

## Experimental design

- Qubits: `3, 4, 5, 6, 7, 8, 9, 10`
- Circuit layers: `2, 3, 4`
- Random seeds: `0, 1, 2, 3, 4`
- Population size: `60`
- Generations: `120`
- Shifted parameters per refinement step: at most `24`
- Total benchmark configurations: `8 × 3 × 5 = 120`
- Simulation: exact Qiskit statevector fidelity

Each method uses the same seed within a qubit-count and circuit-depth configuration, enabling paired comparisons.

## Installation

Python 3.10 or newer is recommended. The archived experiments were executed with Python 3.13.5.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib qiskit qiskit-aer
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running one configuration

The following command executes the five paper methods for 3 qubits, two layers, and seed 0:

```bash
python qfe_icat_2026.py \
  --n-qubits 3 \
  --layers 2 \
  --population 60 \
  --generations 120 \
  --seed 0 \
  --modes FIDELITY_ONLY,QFE_SCORE,FLOW_ONLY,GRADIENT_ONLY,QFE_UNITARY_FLOW \
  --fast-gradient \
  --max-gradient-parameters 24 \
  --save-every 20 \
  --output-dir icat_2026_results/q3_l2_s0
```

## Running the complete grid

Before running the complete paper grid, confirm that the shell script contains:

```bash
QUBITS=(3 4 5 6 7 8 9 10)
LAYERS=(2 3 4)
SEEDS=(0 1 2 3 4)
```

Then run:

```bash
chmod +x run_icat_2026_grid.sh
./run_icat_2026_grid.sh
```

The attached version of `run_icat_2026_grid.sh` currently lists qubits 3–8. Add `9 10` to `QUBITS` to reproduce the complete 120-configuration grid reported in the paper.

## Output files

Each directory follows the pattern `icat_2026_results/q{qubits}_l{layers}_s{seed}/` and contains:

| File or directory | Description |
|---|---|
| `experiment_config.json` | Configuration and execution metadata |
| `history.csv` | Generation-level metrics for all modes |
| `summary_metrics.csv` | Final summary by optimization mode |
| `final_comparison.csv` | Final fidelity, evaluation count, and runtime comparison |
| `threshold_times.csv` | First generation and cost required to reach each fidelity threshold |
| `pairwise_advantage_vs_fidelity_only.csv` | Per-generation differences relative to EFO |
| `plots/` | Fidelity, Functional Information, cost, and advantage plots |

The archive also contains `grid_logs/`, with the console log for each benchmark configuration.

## Results archive

Extract the complete experimental output with:

```bash
unzip icat_2026_results-10qb.zip
```

The archive contains 120 configuration directories and their generation histories, summary tables, comparison tables, plots, execution metadata, and logs. It is approximately 134 MB uncompressed.

## Reproducibility notes

- The GHZ-reachable ansatz is verified at startup by checking that its analytical parameter template reaches unit fidelity.
- Functional Information uses `epsilon = 1e-12` to avoid numerical singularities.
- Base population fidelities are reused within a generation.
- Shifted circuits are evaluated separately.
- No cross-generation evaluation cache is used.
- Runs with more than 24 circuit parameters sample 24 coordinates without replacement for local shifted refinement.
- The original experiment metadata recorded the Python and operating-system versions but not the exact Qiskit patch version.

## Citation

If you use this code or the accompanying results, please cite:

```bibtex
@inproceedings{krause2026qfe,
  author    = {Jonas Krause and Rodrigo Pasti},
  title     = {Quantum Functional Evolution: Information-Shaped Unitary Dynamics for Variational Quantum State Preparation},
  booktitle = {Proceedings of the 8th International Conference on Applied Technologies (ICAT 2026)},
  year      = {2026},
  url       = {https://github.com/jonaskrause/icat2026}
}
```

The citation can be updated with the final Springer CCIS volume, page range, and DOI after publication.

## Authors

- **Jonas Krause** — corresponding author — [jonas.krause1@pucpr.br](mailto:jonas.krause1@pucpr.br)
- **Rodrigo Pasti** — [rodrigo.pasti@pucpr.br](mailto:rodrigo.pasti@pucpr.br)

Postgraduate Program in Smart and Sustainable Cities (PPGCIS), Pontifical Catholic University of Paraná (PUCPR), Curitiba, Brazil.
