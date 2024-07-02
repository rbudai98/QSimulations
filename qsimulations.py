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
        """Periodic jump operators

        Args:
            i (int): Jump operator number
            t (float): Time stamp

        Returns:
            Qobj: Requested jump operator at specified time
        """
        if i == 0:
            return Qobj(-1j * self.H_op() - 0.5 * self.sum_of_V_dag_V(0))
        if i >= 1 and i <= self._system_size_dim:
            return Qobj(
                0.5
                * (
                    self.Pauli_array(self.X, i, self._system_size)
                    - 1j * self.Pauli_array(self.Y, i, self._system_size)
                )
            )
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
            return ret
        for i in np.arange(1, size + 1, 1):
            if i == poz:
                ret = np.kron(ret, op)
            else:
                ret = np.kron(ret, self.I)
        return ret
