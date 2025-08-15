"""
Network Applications for Quantum Characterization, Verification, and Validation

This module extends QCVV tools to quantum network scenarios, providing:
- Protocol certification through Bell inequality tests
- Device-independent security verification
- Multi-node coordination and trust establishment
- Network-wide certification and monitoring
"""

from .protocol_certification import (
    CHSHTest, 
    MerminTestEnhanced, 
    AdaptiveBellTest,
    NetworkCertificationCoordinator
)

__all__ = [
    'CHSHTest',
    'MerminTestEnhanced', 
    'AdaptiveBellTest',
    'NetworkCertificationCoordinator'
]