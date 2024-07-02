# Utility function, system and constants
from scipy.linalg import sqrtm, cosm, sinm, expm
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.linalg as la
import numpy as np
from qutip import *
import math
import qib


class qsimulations:

    _H = 0
    _systemSize = 0
    _nrAncilla = 0
    _nrAncillaDim = 0
    _nrOfDampingOps = 0
    _totalSystemSize = 0
    _totalSystemSizeDim = 0

    I = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Y = np.array([[0.0, -1j], [1j, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])

    creation_op = (X - (1j * Y)) / 2
    annihilation_op = (X + (1j * Y)) / 2

    def __init__(self, H=0, systemSize=0, nrOfDampingOps=0, nrOFAncillas=0):
        """Initialize module parameters

        Args:
            H (Qobj): Hamiltonian of system
            systemSize (int): size of the system
            nrOfDampingOps (int): nr of damping operators
            nrOFAncillas (int): nr of ancilla qubits
        """
        self._H = H
        self._systemSize = systemSize
        self._nrAncilla = nrOFAncillas
        self._nrOfDampingOps = nrOfDampingOps
        self._update_module_varibles()

    def _update_module_varibles(self):
        """Internal function, to update variables based on given parameters"""
        self._systemSizeDim = np.power(
            2, self._systemSize
        )  # Dimensions of the qubit system
        self._nrAncillaDim = np.power(
            2, self._nrAncilla
        )  # Dimensions of ancillary system
        self._totalSystemSize = self._systemSize + self._nrAncilla  # Total system size
        self._totalSystemSizeDim = np.power(
            2, self._totalSystemSize
        )  # Total system dimensions

    def set_H_op(self, H_tmp):
        """Set system Hamiltonian

        Args:
            H_tmp (Qobj): System Hamiltonian
        """
        self._H = H_tmp
        self._update_module_varibles()
        self._prep_energy_states()

    def set_system_size(self, size):
        self._systemSize = size
        self._update_module_varibles()

    def set_nr_of_damping_ops(self, nr):
        self._nrOfDampingOps = nr
        self._update_module_varibles()

    def set_nr_of_ancillas(self, nr):
        self._nrAncilla = nr
        self._update_module_varibles()

    def H_op(self):
        """Time dependent periodic Hamiltonian for TFIM model

        Args:
            t (float): time stamp

        Returns:
            Qobj: Hamiltonian in the requested time
        """
        return Qobj(self._H)

    def V_op(self, i):
        """Damping operators, should be overwritten when system is designed

        Args:
            i (int): Jump operator number

        """
        return 0

    def Pauli_array(self, op, poz, size):
        """Pauli operator on custom system

        Args:
            op (np.array): type of operator defined as numpy array, i.e. X, Y, Z
            poz (int): position of the system the operator acts on, indexing starting from 0
            size (int): size of the total system

        Returns:
            np.array: numpy array of the operator
        """
        ret = 1
        if poz > size:
            return 0
        for i in np.arange(1, size + 1, 1):
            if i == poz:
                ret = np.kron(ret, op)
            else:
                ret = np.kron(ret, self.I)
        return ret

    def outer_prod(self, left_poz, right_poz, size):
        """Outer product of position i and j: |i><j|

        Args:
            left_poz (int): Position of 1 in ket vector, index from 1
            right_poz (int): Position of 1 in the bra vector, index from 1
            size (int): Size of the overall system

        Returns:
            np.array: Requested outer product
        """
        psi_ket = np.array([np.full(size, 0)])
        psi_ket[:, left_poz - 1] = 1.0
        psi_bra = np.array([np.full(size, 0)])
        psi_bra[:, right_poz - 1] = 1.0
        return Qobj(psi_ket.conjugate().T @ psi_bra)

    def commute(self, op1, op2):
        """Commuting operator

        Args:
            op1 (np.array): Operator 1
            op2 (np.array): Operator 2

        Returns:
            np.array: [op1, op2]
        """
        return op1 @ op2 - op2 @ op1

    def anti_commute(self, op1, op2):
        """Anti-commuting operator

        Args:
            op1 (np.array): Operator 1
            op2 (np.array): Operator 2

        Returns:
            np.array: {op1, op2}
        """
        return op1 @ op2 + op2 @ op1

    def _prep_energy_states(self):
        """Prepare pare highest energy level state and ground state"""
        eigenValues, eigenVectors = la.eig(self.H_op().full())
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[idx]

        self.psi_ground = np.matrix(eigenVectors[0].astype(float))
        self.psi_0 = np.matrix(eigenVectors[-1].astype(float))

        self.rho_ground = Qobj(self.psi_ground.conj().T @ self.psi_ground)
        self.rho_0 = Qobj(self.psi_0.conj().T @ self.psi_0)
