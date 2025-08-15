# Quantum Characterization, Verification, and Validation for Network Applications

A comprehensive Python framework for quantum hardware characterization with extensions to **quantum network security verification**. This project demonstrates how traditional QCVV tools naturally extend to distributed quantum systems, providing essential building blocks for quantum network deployment.

## Overview

Quantum networks require robust characterization methods that go beyond single-device QCVV. When quantum hardware moves from isolated laboratory experiments to networked deployments, new certification challenges emerge:

- **Individual device characterization** (traditional QCVV)
- **Network-level security verification** (device-independent protocols)
- **Multi-node coordination and trust establishment**

This project bridges this gap by extending proven QCVV techniques to quantum network scenarios.

---

## Core Capabilities

### 🔬 **Traditional QCVV Foundation**
- **Randomized Benchmarking (RB)**: Multi-qubit and interleaved protocols
- **Entanglement verification**: Witness functions, PPT tests, and negativity analysis  
- **Statistical validation**: Curve fitting, error analysis, and automated calibration
- **Multipartite entanglement**: Bell inequality tests and genuine entanglement detection

### 🌐 **Quantum Network Extensions**
- **Bell inequality certification**: CHSH and Mermin test implementations
- **Device-independent security**: Protocol certification without trusting hardware
- **Network coordination**: Multi-node test scheduling and result aggregation
- **Trust establishment**: Network-wide security parameter calculation

### 🛠️ **Technical Infrastructure**
- **Modular architecture**: Core QCVV + network applications
- **Comprehensive testing**: Unit tests with continuous integration
- **Interactive demonstrations**: Jupyter notebooks for both single-device and network scenarios

---

## Project Structure

```
QCVV/
├── src/
│   ├── state_utils.py           # Quantum states, entanglement witness
│   ├── entanglement_analysis.py  # Batch analysis, multipartite tools
│   ├── symbolic_utils.py        # Symbolic computation utilities
│   └── network_applications/    # Network certification tools
│       ├── protocol_certification.py  # Bell tests, device-independent security
│       └── __init__.py
├── notebooks/
│   ├── Multipartite_Entanglement.ipynb  # Core QCVV demonstrations
│   ├── RB.ipynb                         # Randomized benchmarking
│   ├── witness.ipynb                    # Entanglement witness functions
│   └── network_analysis/                # ⭐ Featured: Multi-scale network analysis
│       ├── Quantum_Network_Demo.ipynb           # Three-layer European scenario
│       ├── Quantum_Network_Integrated_Toolkit.ipynb  # QRC+OQS+QCVV integration
│       └── README.md                    # Detailed analysis documentation
├── tests/                       # Comprehensive unit tests
└── docs/                       # Technical documentation
```

## Code Modules

| Module | Description | Network Relevance |
|--------|-------------|-------------------|
| **Core QCVV** |
| `state_utils.py` | Quantum state generation, entropy, witness functions | Provides Bell state preparation |
| `entanglement_analysis.py` | Batch analysis, all-bipartition tools | Extends to multi-node entanglement |
| `symbolic_utils.py` | Symbolic quantum computation tools | Theoretical protocol analysis |
| **Network Applications** |
| `protocol_certification.py` | Bell inequality tests (CHSH, Mermin), adaptive selection | **Direct quantum network security** |
| `NetworkCertificationCoordinator` | Multi-node test coordination and trust assessment | **Quantum internet deployment** |

---

## Key Features & Applications

### 🔐 **Device-Independent Network Security**
```python
# Example: Network-wide Bell inequality certification
coordinator = NetworkCertificationCoordinator()
coordinator.add_node("Alice", {"qubits": 2, "fidelity": 0.95})
coordinator.add_node("Bob", {"qubits": 3, "fidelity": 0.92})
coordinator.add_connection("Alice", "Bob")

# Automatic test selection and execution
results = coordinator.run_network_certification('standard')
# → Network certified: ✅ CHSH violation detected, security parameter: 0.389
```

### 📊 **Adaptive Bell Test Selection**
- **2-qubit systems**: Automatic CHSH inequality testing (industry standard)
- **N≥3 systems**: Enhanced Mermin inequality with higher violation potential  
- **Mixed networks**: Intelligent test selection based on node capabilities

### 🌐 **Network Topology Management**
- Node capability tracking (qubit count, fidelity, connectivity)
- Pairwise test coordination across arbitrary network topologies
- Trust score calculation and network-wide security assessment

---

## Theoretical Foundations

This project builds on fundamental advances in quantum information theory and network security protocols. The verification tools implemented here leverage well-established principles in quantum cryptography and distributed system certification:

- **Device-Independent Security**: Bell inequality violation serves as a hardware-agnostic security guarantee [1,2]
- **Multipartite Entanglement**: Extension of bipartite Bell tests to N-party systems for scalable verification [3,4]
- **Network-Scale Certification**: Distributed coordination protocols for establishing trust across quantum networks [5,6]

### Core Research Areas
- **Bell Inequality Theory**: Implementation of CHSH and Mermin inequalities for security certification [7,8]
- **Entanglement Verification**: PPT tests, negativity analysis, and witness functions for multipartite systems [9,10]
- **Quantum Network Topology**: Analysis of decoherence propagation and routing optimization [11,12]

---

## 🚀 **Featured: Quantum Network Analysis**

This repository includes comprehensive **multi-scale quantum network analysis** demonstrating practical applications of QCVV principles to real-world scenarios:

### 📊 **Problem-Solving Framework**
- **[Network Demo](notebooks/network_analysis/Quantum_Network_Demo.ipynb)**: Three-layer analysis (decoherence, routing, security) on European quantum network scenario
- **[Integrated Toolkit](notebooks/network_analysis/Quantum_Network_Integrated_Toolkit.ipynb)**: Multi-toolbox integration demonstrating QRC + OQS + QCVV synergy

### 🎯 **Key Demonstrations**
1. **Topology Optimization**: Comparing linear vs star networks under realistic decoherence models
2. **Adaptive Routing**: Traffic prediction and QoS-aware resource allocation 
3. **End-to-End Security**: Network-wide Bell inequality certification and trust scoring

### 📈 **Technical Depth**
- **Cross-layer integration**: Physical layer constraints → routing decisions → security verification
- **Scalability analysis**: European 4-city network with realistic parameters
- **Quantitative metrics**: Performance evaluation with actionable insights

**➡️ [Explore Network Analysis](notebooks/network_analysis/)**

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [repository-url]
   cd quantum-error-correction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run basic tests:**
   ```bash
   python src/network_applications/protocol_certification.py
   ```

4. **Explore interactive demos:**
   ```bash
   jupyter lab notebooks/
   ```

---

## Example Results

### Single Bell Test
```
CHSH Test (Bell State): 2.389 > 2.0 ✅
Security Parameter: 0.389
Device-Independent Security: CERTIFIED
```

### Network Certification
```
🔐 NETWORK CERTIFICATION REPORT
📊 Network Status: ✅ CERTIFIED  
🔗 Total Connections: 4
✅ Passed: 4 | ❌ Failed: 0
🏆 Network Trust Score: 0.389
```

---

## About This Project

This repository represents independent research and prototyping in quantum network security. The goal is to demonstrate how traditional quantum hardware characterization naturally extends to distributed quantum systems, providing essential tools for the emerging quantum internet.

**Technical Focus**: From single-device QCVV to network-scale verification  
**Research Value**: Bridges laboratory experiments and real-world quantum network deployment  
**Future Integration**: Designed as potential verification layer for quantum network simulators and testbeds

---

## References

[1] Quantum-journal.org. "Device-independent quantum key distribution from generalized CHSH inequalities." *Quantum* 5, 444 (2021). [https://quantum-journal.org/papers/q-2021-04-26-444/](https://quantum-journal.org/papers/q-2021-04-26-444/)

[2] arXiv:2502.12241. "Certification of quantum correlations and DIQKD at arbitrary distances through routed Bell tests." (2025).

[3] Quantum-journal.org. "Randomness versus nonlocality in the Mermin-Bell experiment with three parties." *Quantum* 2, 82 (2018). [https://quantum-journal.org/papers/q-2018-08-17-82/](https://quantum-journal.org/papers/q-2018-08-17-82/)

[4] arXiv:2101.10307. "Testing Scalable Bell Inequalities for Quantum Graph States on IBM Quantum Devices." (2021).

[5] arXiv:2411.07410. "Control Protocol for Entangled Pair Verification in Quantum Optical Networks." (2024).

[6] arXiv:2503.16383. "Quantum Characterization, Verification, and Validation." (2025).

[7] Springer. "Construction of Bell inequalities based on the CHSH structure." *Quantum Information Processing* 16, 1562 (2017). [https://link.springer.com/article/10.1007/s11128-017-1562-6](https://link.springer.com/article/10.1007/s11128-017-1562-6)

[8] Quantum-journal.org. "Device-independent quantum key distribution with asymmetric CHSH inequalities." *Quantum* 5, 443 (2021). [https://quantum-journal.org/papers/q-2021-04-26-443/](https://quantum-journal.org/papers/q-2021-04-26-443/)

[9] arXiv:1410.7094. "Computable entanglement conversion witness that is better than the negativity." (2014).

[10] *Scientific Reports*. "Enhancing collective entanglement witnesses through correlation with state purity." 14, 65385 (2024). [https://www.nature.com/articles/s41598-024-65385-7](https://www.nature.com/articles/s41598-024-65385-7)

[11] arXiv:2208.01983. "Valid and efficient entanglement verification with finite copies of a quantum state." (2022).

[12] arXiv:2407.09942. "Benchmarking quantum gates and circuits." (2024).