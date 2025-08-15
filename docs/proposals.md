## Project A: Randomized Benchmarking (RB) and Entanglement Verification — Theory and Implementation

### Scope:

SPAM-robust RB, multi-qubit/interleaved RB, automated RB calibration, comparison with QPT/GST.
Integration with multipartite entanglement verification and witnesses (e.g., Mermin inequality, concurrence, spin squeezing).
Lindblad equation-based calculations of entanglement entropy, Schmidt spectrum, and witness expectation, exploring the impact of noise/decoherence on verification.

### Key Points:

- Theory, simulation, and automation of RB methods.
- Design and verification of entanglement witnesses, correlated with RB results.
- Comparison of RB, QPT, and GST in terms of results and resource requirements.
- Entanglement verification and witness decay under Lindblad dynamics.

### Subtasks:

- SPAM noise model design and simulation; data comparison between RB fitting and direct fidelity estimation
- Automated RB calibration loop (adjusting noise model parameters based on RB results)
- Simulation and analysis of multi-qubit and interleaved RB
- Correlation analysis between RB results and quantum control/calibration parameters
- Resource and result comparison between RB and QPT/GST


## Project B: Convex Optimization (SDP) and Machine Learning for Entanglement Witnesses and Quantum State Classification

### Scope:
Automated design of entanglement witnesses using SDP (e.g., PPT check, optimal witness).
Machine learning for quantum state classification, automatic witness generation, and anomaly detection.
Integration with Lindblad/open system dynamics to design the most robust witnesses and predict entanglement lifetime.

### Key Points:

- Application of CVXPY/Qiskit SDP tools for multipartite entanglement witnesses.
- Construction of quantum state classifiers and anomaly detectors using sklearn/torch.
- Data-driven QCVV protocols.
- Witness optimization and automated diagnostics in open quantum systems.

### Subtasks:

- Anomaly detection for quantum experimental data (e.g., decoherence, noise)
- Classification and verification of complex multipartite entangled systems
- Quantum device optimization and automated diagnostics
- Machine learning prediction of witness decay and entanglement lifetime under Lindblad dynamics

## Project C: Quantum Networks, Graph Theory, and Multipartite Entanglement Distribution

### Scope:
Entanglement distribution in quantum networks; partitioning and entropy distribution of GHZ/W/Dicke states across network nodes.
Graph/hypergraph structure analysis using networkx; design of graph-based QCVV protocols.
Exploration of graph theory in quantum network connectivity, entanglement distribution, and MBQC/Cluster State roles.

### Key Points:

- Simulation and verification of entanglement distribution in quantum networks.
- Graph/hypergraph theory for multipartite entanglement structure analysis.
- Design of graph-based quantum computation/communication protocols.

### Subtasks:

- Design and verification of QCVV protocols based on GHZ/W/Dicke states
- MBQC computation flow design and visualization (e.g., effect of nonlocal unitary transformations on graph states)
- Exploration of graph theory in quantum network connectivity and entanglement distribution

## Project D: Causal Inference, Non-Markovianity, and Multi-Step QCVV Protocols

### Scope:
Causal structure (causal hierarchy), process tensor formalism for analyzing the impact of non-Markovianity on multipartite entanglement.
Design and verification of multi-step QCVV protocols; witness optimization under non-Markovianity.

### Key Points:

- Theory and simulation of process tensors.
- Entanglement verification and protocol design under non-Markovianity.
- Application of causal inference in QCVV.

### Subtasks:

- Design and verification of multi-step QCVV protocols
- Witness optimization under non-Markovianity