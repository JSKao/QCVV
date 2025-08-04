# Quantum Characterization, Verification, and Validation (QCVV) — Personal Learning Notes

This repository is a personal learning and prototyping project focused on quantum hardware characterization (QCVV). The content is mainly in Python/Jupyter notebooks, covering topics such as Randomized Benchmarking (RB), SPAM robustness, automated calibration, and initial explorations of AI/ML concepts.

The current focus is on simulation and numerical experiments. Code structure and features are still evolving and expanding.

## Project Structure

- `notebooks/` — Main Jupyter notebooks for experiments, simulations, and analysis
- `src/` — Reusable QCVV Python functions/modules
- `data/` — Simulation results, raw data, plots
- `docs/` — Documentation, project summaries, technical notes
- `tests/` — Unit tests

## Code Modules

| Module            | Description                                      |
|-------------------|--------------------------------------------------|
| state_utils.py    | State generation, entropy, spectrum, witness     |
| batch_analysis.py | Batch/global analysis, all-bipartition tools     |
| symbolic_utils.py | Symbolic (sympy) state and entanglement tools    |

> For detailed function classification, see the docstrings at the top of each file or the docs/ directory.

## Current Topics

- Randomized Benchmarking (RB), SPAM robustness
- RB EPC vs. direct fidelity estimation
- Automated calibration and closed-loop optimization prototypes
- Multi-qubit and interleaved RB simulations
- Correlation between RB results and quantum control parameters
- Comparison of RB with QPT/GST and other methods
- Initial applications of AI/ML concepts to quantum calibration

## About

All code and documentation are the result of my independent study, research, and prototyping. The goal is to bring modern software engineering and data-driven thinking into quantum hardware characterization and validation workflows.

### Future Plans (Work in Progress)
- Gradually expand to include multipartite entanglement verification, SDP/machine learning, graph theory, and causal inference modules
- See docs/proposals.md for more details on planned directions