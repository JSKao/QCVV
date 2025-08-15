"""
Quantum Protocol Certification for Network Applications

This module provides tools for certifying quantum communication protocols
through Bell inequality tests, enabling device-independent security verification
in quantum networks.

Key Features:
- CHSH inequality testing (2-qubit systems)
- Mermin inequality testing (N≥3 qubit systems) 
- Adaptive test selection based on system capabilities
- Network-level certification coordination
- Statistical analysis and security parameter extraction
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import warnings

# Import existing QCVV tools
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state_utils import ghz_state, mermin_operator, mermin_lhv_bound, check_mermin_witness


class CHSHTest:
    """
    CHSH inequality test for 2-qubit systems.
    
    The CHSH (Clauser-Horne-Shimony-Holt) inequality is the most widely used
    Bell test in practical quantum cryptography applications.
    """
    
    def __init__(self, measurement_angles: Optional[Dict[str, List[float]]] = None):
        """
        Initialize CHSH test with optimal measurement angles.
        
        Args:
            measurement_angles: Custom angles for Alice/Bob measurements
                              Default uses optimal angles for maximum violation
        """
        if measurement_angles is None:
            # Optimal angles for maximum CHSH violation
            self.angles = {
                'alice': [0, np.pi/4],          # a0=0°, a1=45°
                'bob': [np.pi/8, 3*np.pi/8]     # b0=22.5°, b1=67.5°
            }
        else:
            self.angles = measurement_angles
            
        self.classical_bound = 2.0
        self.quantum_bound = 2 * np.sqrt(2)  # ≈ 2.828
        
    def pauli_measurement(self, state: np.ndarray, angle: float, qubit: int) -> float:
        """
        Simulate Pauli measurement at given angle on specified qubit.
        
        Args:
            state: 2-qubit quantum state (4D vector)
            angle: Measurement angle in radians
            qubit: Which qubit to measure (0 for Alice, 1 for Bob)
            
        Returns:
            Expectation value of measurement
        """
        # Pauli operators
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Measurement operator: cos(θ)σ_z + sin(θ)σ_x
        local_op = np.cos(angle) * sigma_z + np.sin(angle) * sigma_x
        
        # Extend to 2-qubit operator
        I = np.eye(2)
        if qubit == 0:  # Alice's measurement
            measurement_op = np.kron(local_op, I)
        else:  # Bob's measurement
            measurement_op = np.kron(I, local_op)
            
        # Calculate expectation value
        rho = np.outer(state, state.conj())
        expectation = np.real(np.trace(measurement_op @ rho))
        
        return expectation
    
    def correlation_function(self, state: np.ndarray, alice_angle: float, bob_angle: float) -> float:
        """
        Calculate correlation function E(a,b) = <A_a ⊗ B_b>.
        
        Args:
            state: 2-qubit quantum state
            alice_angle: Alice's measurement angle
            bob_angle: Bob's measurement angle
            
        Returns:
            Correlation value E(a,b)
        """
        # Pauli operators
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Local measurement operators
        A = np.cos(alice_angle) * sigma_z + np.sin(alice_angle) * sigma_x
        B = np.cos(bob_angle) * sigma_z + np.sin(bob_angle) * sigma_x
        
        # Joint measurement operator
        AB = np.kron(A, B)
        
        # Calculate correlation
        rho = np.outer(state, state.conj())
        correlation = np.real(np.trace(AB @ rho))
        
        return correlation
    
    def chsh_value(self, state: np.ndarray) -> float:
        """
        Calculate CHSH value for given quantum state.
        
        Args:
            state: 2-qubit quantum state
            
        Returns:
            CHSH value S = |E(a0,b0) - E(a0,b1) + E(a1,b0) + E(a1,b1)|
        """
        a0, a1 = self.angles['alice']
        b0, b1 = self.angles['bob']
        
        # Calculate all four correlation functions
        E_a0_b0 = self.correlation_function(state, a0, b0)
        E_a0_b1 = self.correlation_function(state, a0, b1)
        E_a1_b0 = self.correlation_function(state, a1, b0)
        E_a1_b1 = self.correlation_function(state, a1, b1)
        
        # CHSH combination
        S = abs(E_a0_b0 - E_a0_b1 + E_a1_b0 + E_a1_b1)
        
        return S
    
    def test_bell_violation(self, state: np.ndarray) -> Dict[str, Union[float, bool]]:
        """
        Test Bell inequality violation for device-independent certification.
        
        Args:
            state: 2-qubit quantum state to test
            
        Returns:
            Dictionary with test results and security assessment
        """
        S = self.chsh_value(state)
        
        # Classical threshold (security requirement)
        classical_violation = S > self.classical_bound
        
        # Quantum advantage assessment
        quantum_advantage = (S - self.classical_bound) / (self.quantum_bound - self.classical_bound)
        
        # Security parameter (rough estimate)
        # Higher violation → better security against classical eavesdropping
        security_parameter = max(0, S - self.classical_bound)
        
        return {
            'chsh_value': S,
            'classical_bound': self.classical_bound,
            'quantum_bound': self.quantum_bound,
            'violates_classical': classical_violation,
            'quantum_advantage_ratio': quantum_advantage,
            'security_parameter': security_parameter,
            'certification_passed': classical_violation
        }


class MerminTestEnhanced:
    """
    Enhanced Mermin inequality test building on existing QCVV code.
    
    Extends the existing mermin_witness functionality for network applications.
    """
    
    def __init__(self, max_qubits: int = 5):
        """
        Initialize enhanced Mermin test.
        
        Args:
            max_qubits: Maximum number of qubits to support
        """
        self.max_qubits = max_qubits
        self.supported_sizes = [3, 4, 5]  # Based on existing implementation
        
    def test_bell_violation(self, state: np.ndarray, N: int) -> Dict[str, Union[float, bool]]:
        """
        Test Mermin inequality violation using existing QCVV infrastructure.
        
        Args:
            state: N-qubit quantum state
            N: Number of qubits
            
        Returns:
            Dictionary with test results
        """
        if N not in self.supported_sizes:
            raise ValueError(f"N={N} not supported. Use N in {self.supported_sizes}")
            
        # Use existing functions - calculate Mermin expectation directly
        rho = np.outer(state, state.conj())
        M = mermin_operator(N)
        mermin_val = np.real(np.trace(M @ rho))  # Direct Mermin expectation
        lhv_bound = mermin_lhv_bound(N)
        
        # Assessment
        classical_violation = mermin_val > lhv_bound
        violation_strength = mermin_val - lhv_bound
        
        # Theoretical maximum for this N (GHZ states achieve this)
        theoretical_max = 2 ** (N-1) if N % 2 == 1 else 2 ** (N//2)  # Corrected formula
        quantum_advantage = violation_strength / (theoretical_max - lhv_bound) if theoretical_max > lhv_bound else 0
        
        return {
            'mermin_value': mermin_val,
            'lhv_bound': lhv_bound,
            'theoretical_maximum': theoretical_max,
            'violation_strength': violation_strength,
            'violates_classical': classical_violation,
            'quantum_advantage_ratio': quantum_advantage,
            'security_parameter': max(0, violation_strength),
            'certification_passed': classical_violation
        }


class AdaptiveBellTest:
    """
    Adaptive Bell test selector based on system capabilities.
    
    Automatically chooses the most appropriate Bell test based on:
    - Number of available qubits
    - Required security level
    - Experimental constraints
    """
    
    def __init__(self):
        self.chsh_test = CHSHTest()
        self.mermin_test = MerminTestEnhanced()
        
    def recommend_test(self, num_qubits: int, security_requirement: str = 'standard') -> str:
        """
        Recommend optimal Bell test for given system.
        
        Args:
            num_qubits: Available qubits in the system
            security_requirement: 'basic', 'standard', or 'high'
            
        Returns:
            Recommended test type
        """
        if num_qubits < 2:
            raise ValueError("At least 2 qubits required for Bell tests")
        elif num_qubits == 2:
            return 'CHSH'
        elif num_qubits >= 3:
            if security_requirement == 'basic':
                return 'CHSH'  # Simpler, more robust
            else:
                return 'Mermin'  # Higher violation potential
        
    def run_optimal_test(self, state: np.ndarray, num_qubits: int, 
                        security_requirement: str = 'standard') -> Dict:
        """
        Run the optimal Bell test for given parameters.
        
        Args:
            state: Quantum state to test
            num_qubits: Number of qubits
            security_requirement: Security level requirement
            
        Returns:
            Test results with metadata
        """
        recommended_test = self.recommend_test(num_qubits, security_requirement)
        
        if recommended_test == 'CHSH':
            if num_qubits > 2:
                # Extract 2-qubit subsystem for CHSH test
                state_2q = self._extract_two_qubit_subsystem(state, num_qubits)
                results = self.chsh_test.test_bell_violation(state_2q)
            else:
                results = self.chsh_test.test_bell_violation(state)
            results['test_type'] = 'CHSH'
            
        else:  # Mermin
            results = self.mermin_test.test_bell_violation(state, num_qubits)
            results['test_type'] = 'Mermin'
            
        results['recommendation_reason'] = f"{recommended_test} optimal for {num_qubits} qubits, {security_requirement} security"
        
        return results
    
    def _extract_two_qubit_subsystem(self, state: np.ndarray, N: int) -> np.ndarray:
        """Extract 2-qubit subsystem from larger state for CHSH testing."""
        # Simple approach: trace out all but first two qubits
        # For more sophisticated selection, could analyze entanglement structure
        
        state_tensor = state.reshape([2] * N)
        # Trace out qubits 2, 3, ..., N-1
        for i in range(N-1, 1, -1):
            state_tensor = np.trace(state_tensor, axis1=0, axis2=i)
            
        return state_tensor.flatten()


class NetworkCertificationCoordinator:
    """
    Coordinates Bell inequality tests across a quantum network.
    
    Manages multi-node certification by:
    - Tracking network topology and node capabilities
    - Coordinating pairwise Bell tests
    - Aggregating results for network-wide trust assessment
    - Handling test failures and retries
    """
    
    def __init__(self):
        self.adaptive_tester = AdaptiveBellTest()
        self.network_topology = {}  # {node_id: {connections: [], capabilities: {}}}
        self.certification_history = {}  # {(node1, node2): [test_results]}
        
    def add_node(self, node_id: str, capabilities: Dict[str, Union[int, float]]):
        """
        Add a node to the network.
        
        Args:
            node_id: Unique identifier for the node
            capabilities: Node capabilities (e.g., {'qubits': 2, 'fidelity': 0.95})
        """
        self.network_topology[node_id] = {
            'connections': [],
            'capabilities': capabilities,
            'status': 'active'
        }
        
    def add_connection(self, node1: str, node2: str):
        """
        Add a quantum connection between two nodes.
        
        Args:
            node1, node2: Node identifiers to connect
        """
        if node1 not in self.network_topology or node2 not in self.network_topology:
            raise ValueError(f"Both nodes must be added before connecting")
            
        # Add bidirectional connection
        self.network_topology[node1]['connections'].append(node2)
        self.network_topology[node2]['connections'].append(node1)
        
        # Initialize certification history for this pair
        pair_key = tuple(sorted([node1, node2]))
        if pair_key not in self.certification_history:
            self.certification_history[pair_key] = []
            
    def get_connected_pairs(self) -> List[Tuple[str, str]]:
        """Get all connected node pairs in the network."""
        pairs = []
        processed = set()
        
        for node1, info in self.network_topology.items():
            for node2 in info['connections']:
                pair = tuple(sorted([node1, node2]))
                if pair not in processed:
                    pairs.append(pair)
                    processed.add(pair)
                    
        return pairs
    
    def get_pair_capabilities(self, node1: str, node2: str) -> Dict[str, int]:
        """Get combined capabilities for a node pair."""
        cap1 = self.network_topology[node1]['capabilities']
        cap2 = self.network_topology[node2]['capabilities']
        
        # Take minimum capabilities for conservative testing
        combined_qubits = min(cap1.get('qubits', 2), cap2.get('qubits', 2))
        
        return {
            'qubits': combined_qubits,
            'node1_qubits': cap1.get('qubits', 2),
            'node2_qubits': cap2.get('qubits', 2)
        }
    
    def simulate_pair_test(self, node1: str, node2: str, 
                          security_requirement: str = 'standard') -> Dict:
        """
        Simulate Bell test between two connected nodes.
        
        Args:
            node1, node2: Nodes to test
            security_requirement: Security level requirement
            
        Returns:
            Test results with node pair metadata
        """
        pair_caps = self.get_pair_capabilities(node1, node2)
        num_qubits = pair_caps['qubits']
        
        # Generate appropriate test state based on capabilities
        if num_qubits == 2:
            # Bell state for CHSH
            test_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        elif num_qubits >= 3:
            # GHZ state for Mermin
            test_state = ghz_state(num_qubits)
        else:
            raise ValueError(f"Insufficient qubits for testing: {num_qubits}")
            
        # Run adaptive test
        results = self.adaptive_tester.run_optimal_test(
            test_state, num_qubits, security_requirement
        )
        
        # Add metadata
        results.update({
            'node_pair': (node1, node2),
            'test_timestamp': np.random.randint(1000000, 9999999),  # Simulate timestamp
            'pair_capabilities': pair_caps
        })
        
        return results
    
    def run_network_certification(self, security_requirement: str = 'standard') -> Dict:
        """
        Run comprehensive Bell tests across all connected node pairs.
        
        Args:
            security_requirement: Network-wide security requirement
            
        Returns:
            Network certification report
        """
        print("🔍 Starting network-wide Bell inequality certification...")
        
        all_pairs = self.get_connected_pairs()
        pair_results = {}
        network_stats = {
            'total_pairs': len(all_pairs),
            'passed_pairs': 0,
            'failed_pairs': 0,
            'average_security_parameter': 0.0,
            'min_security_parameter': float('inf'),
            'network_trust_score': 0.0
        }
        
        security_parameters = []
        
        for node1, node2 in all_pairs:
            print(f"  Testing {node1} ↔ {node2}...")
            
            try:
                results = self.simulate_pair_test(node1, node2, security_requirement)
                pair_key = tuple(sorted([node1, node2]))
                pair_results[pair_key] = results
                
                # Store in history
                self.certification_history[pair_key].append(results)
                
                # Update statistics
                if results['certification_passed']:
                    network_stats['passed_pairs'] += 1
                    security_param = results['security_parameter']
                    security_parameters.append(security_param)
                    network_stats['min_security_parameter'] = min(
                        network_stats['min_security_parameter'], security_param
                    )
                else:
                    network_stats['failed_pairs'] += 1
                    print(f"    ❌ Certification FAILED for {node1}-{node2}")
                    
            except Exception as e:
                print(f"    🚨 Error testing {node1}-{node2}: {e}")
                network_stats['failed_pairs'] += 1
                
        # Calculate network-wide metrics
        if security_parameters:
            network_stats['average_security_parameter'] = np.mean(security_parameters)
            # Network trust score: percentage of passed pairs weighted by min security
            pass_rate = network_stats['passed_pairs'] / network_stats['total_pairs']
            min_security = network_stats['min_security_parameter']
            network_stats['network_trust_score'] = pass_rate * min(min_security, 1.0)
        else:
            network_stats['min_security_parameter'] = 0.0
            
        # Overall network certification
        network_certified = (network_stats['failed_pairs'] == 0 and 
                           network_stats['passed_pairs'] > 0)
        
        return {
            'network_certified': network_certified,
            'statistics': network_stats,
            'pair_results': pair_results,
            'security_requirement': security_requirement,
            'topology_summary': {
                'nodes': list(self.network_topology.keys()),
                'total_connections': len(all_pairs)
            }
        }
    
    def print_network_report(self, certification_results: Dict):
        """Print formatted network certification report."""
        print("\n" + "="*60)
        print("🔐 QUANTUM NETWORK CERTIFICATION REPORT")
        print("="*60)
        
        stats = certification_results['statistics']
        certified = certification_results['network_certified']
        
        print(f"📊 Network Status: {'✅ CERTIFIED' if certified else '❌ NOT CERTIFIED'}")
        print(f"🔗 Total Connections: {stats['total_pairs']}")
        print(f"✅ Passed: {stats['passed_pairs']}")
        print(f"❌ Failed: {stats['failed_pairs']}")
        print(f"🛡️  Average Security: {stats['average_security_parameter']:.4f}")
        print(f"🔒 Min Security: {stats['min_security_parameter']:.4f}")
        print(f"🏆 Network Trust Score: {stats['network_trust_score']:.4f}")
        
        print("\n📋 Individual Pair Results:")
        for pair_key, results in certification_results['pair_results'].items():
            node1, node2 = pair_key
            status = "✅" if results['certification_passed'] else "❌"
            test_type = results['test_type']
            security = results['security_parameter']
            print(f"  {status} {node1}-{node2}: {test_type} test, security={security:.4f}")
            
        print("="*60)


if __name__ == "__main__":
    # Example usage and testing
    print("=== Quantum Protocol Certification Test ===")
    
    # Test CHSH with Bell state
    print("\n1. CHSH Test with Bell State")
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |00⟩ + |11⟩
    
    chsh = CHSHTest()
    chsh_results = chsh.test_bell_violation(bell_state)
    
    print(f"CHSH value: {chsh_results['chsh_value']:.4f}")
    print(f"Classical bound: {chsh_results['classical_bound']}")
    print(f"Violates classical bound: {chsh_results['violates_classical']}")
    print(f"Security parameter: {chsh_results['security_parameter']:.4f}")
    
    # Test Mermin with GHZ state
    print("\n2. Mermin Test with GHZ State")
    ghz_3 = ghz_state(3)
    
    mermin = MerminTestEnhanced()
    mermin_results = mermin.test_bell_violation(ghz_3, 3)
    
    print(f"Mermin value: {mermin_results['mermin_value']:.4f}")
    print(f"LHV bound: {mermin_results['lhv_bound']}")
    print(f"Violates classical bound: {mermin_results['violates_classical']}")
    print(f"Security parameter: {mermin_results['security_parameter']:.4f}")
    
    # Test adaptive selection
    print("\n3. Adaptive Test Selection")
    adaptive = AdaptiveBellTest()
    
    adaptive_results = adaptive.run_optimal_test(ghz_3, 3, 'standard')
    print(f"Recommended test: {adaptive_results['test_type']}")
    print(f"Reason: {adaptive_results['recommendation_reason']}")
    print(f"Certification passed: {adaptive_results['certification_passed']}")
    
    # Test network coordination
    print("\n4. Network Certification Coordination")
    coordinator = NetworkCertificationCoordinator()
    
    # Build a test network
    coordinator.add_node("Alice", {"qubits": 2, "fidelity": 0.95})
    coordinator.add_node("Bob", {"qubits": 3, "fidelity": 0.92})
    coordinator.add_node("Charlie", {"qubits": 2, "fidelity": 0.88})
    coordinator.add_node("David", {"qubits": 4, "fidelity": 0.90})
    
    # Add connections
    coordinator.add_connection("Alice", "Bob")
    coordinator.add_connection("Bob", "Charlie")
    coordinator.add_connection("Charlie", "David")
    coordinator.add_connection("Alice", "David")  # Create a cycle
    
    # Run network-wide certification
    network_results = coordinator.run_network_certification('standard')
    
    # Print detailed report
    coordinator.print_network_report(network_results)