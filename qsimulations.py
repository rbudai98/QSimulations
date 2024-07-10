# Utility function, system and constants
import scipy.linalg as la
import numpy as np
from qutip import *
import math

ket_0 = np.array([[1.0, 0.0]]).astype(complex)
ket_1 = np.array([[0.0, 1.0]]).astype(complex)

I = np.array([[1.0, 0.0], [0.0, 1.0]]).astype(complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]]).astype(complex)
Y = np.array([[0.0, -1j], [1j, 0.0]]).astype(complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]]).astype(complex)

creation_op = (X - (1j * Y)) / 2
annihilation_op = (X + (1j * Y)) / 2


def Pauli_array(op, poz, size):
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
            ret = np.kron(ret, I)
    return ret


def outer_prod(left_poz, right_poz, size):
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
    return Qobj(psi_ket.conjugate().T.astype(complex) @ psi_bra.astype(complex))


def commute(op1, op2):
    """Commuting operator

    Args:
        op1 (np.array): Operator 1
        op2 (np.array): Operator 2

    Returns:
        np.array: [op1, op2]
    """
    return op1 @ op2 - op2 @ op1


def anti_commute(op1, op2):
    """Anti-commuting operator

    Args:
        op1 (np.array): Operator 1
        op2 (np.array): Operator 2

    Returns:
        np.array: {op1, op2}
    """
    return op1 @ op2 + op2 @ op1


def Taylor_approximtion(Operator, order, dt, initial_state):
    """Taylor approximation for dilated hamiltonian

    Args:
        nrAncillaDim (int): ancillary system dimension
        systemSizeDim (int): system dimension
        Operator (Qobj): Hamiltonian to be aproximated according to exp(sqrt(dt))
        order (_type_): _description_
        dt (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i in np.arange(1, order + 1, 1):
        tmpVal = Operator
        for _ in np.arange(2, i + 1, 1):
            tmpVal = tmpVal @ Operator
        initial_state = (
            initial_state + np.power(-1j * dt, i) / math.factorial(i) * tmpVal
        )
    return initial_state


class qsimulations:

    _systemSize = 0
    _nrAncilla = 0
    _nrAncillaDim = 0
    _nrOfDampingOps = 0
    _totalSystemSize = 0
    _totalSystemSizeDim = 0

    def __init__(self, systemSize=0, nrOfDampingOps=0, nrOFAncillas=0):
        """Initialize module parameters

        Args:
            H (Qobj): Hamiltonian of system
            systemSize (int): size of the system
            nrOfDampingOps (int): nr of damping operators
            nrOFAncillas (int): nr of ancilla qubits
        """
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

    def set_system_size(self, size):
        self._systemSize = size
        self._update_module_varibles()

    def set_nr_of_damping_ops(self, nr):
        self._nrOfDampingOps = nr
        self._update_module_varibles()

    def set_nr_of_ancillas(self, nr):
        self._nrAncilla = nr
        self._update_module_varibles()

    def H_op(t=0):
        """Time dependent periodic Hamiltonian for TFIM model

        Args:
            t (float): time stamp

        Returns:
            Qobj: Hamiltonian in the requested time
        """
        return Qobj(0)

    def V_op(i, t=0):
        """Damping operators, should be overwritten when system is designed

        Args:
            i (int): Jump operator number

        """
        return Qobj(0)

    def _prep_energy_states(self, t=0):
        """Prepare pare highest energy level state and ground state"""
        eigenValues, eigenVectors = la.eig(self.H_op(t).full())
        self._systemSize = (int)(np.round(np.sqrt(np.size(self.H_op(t).full()[:, 0]))))
        self._update_module_varibles()

        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[idx]

        self.psi_ground = np.array([eigenVectors[0].astype(complex)])
        self.psi_0 = np.array([eigenVectors[-1].astype(complex)])

        self.rho_ground = Qobj(self.psi_ground.conj().T @ self.psi_ground)
        self.rho_0 = Qobj(self.psi_0.conj().T @ self.psi_0)

    def H_op_derivative(t=0):
        """Time derivative of periodic Hamiltonian

        Args:
            t (float): time stamp

        Returns:
            np.array: The drived Hamiltonian n requested time stamp
        """
        return Qobj(0)

    def V_op_derivative(i, t=0):
        """Time derivative of jump operators

        Args:
            i (int): Nr of jump operator
            t (float): Time stamp

        Returns:
            Qobj: Time derivative of the selected jump operator
        """
        return Qobj(0)

    def sum_of_V_dag_V(self):
        """Summation of jump operators

        Args:
            t (float): Time stamp

        Returns:
            Qobj: Sum of all jump operators
        """
        sum = 0
        for j in np.arange(1, self._nrOfDampingOps + 1, 1):
            sum = sum + self.V_op(j).full().conj().T @ self.V_op(j).full()
        return Qobj(sum)

    def H_tilde_first_order(self, dt, t=0):
        """Form of H tilde:
        H =
        sqrt(dt)H   |  V_1^{\\dag}   |   V_2^{\\dag}  |       0
                V_1     |       0       |       0       |       0
                V_2     |       0       |       0       |       0
                0       |       0       |       0       |       0

                Args:
                t (float): time stamp
        """
        sum = Qobj(
            np.kron(
                outer_prod(0, 0, self._nrAncillaDim).full(),
                (np.sqrt(dt) * self.H_op(t)).full(),
            )
        )
        # First Row
        for j in np.arange(1, self._nrOfDampingOps + 1, 1):
            sum = sum + Qobj(
                np.kron(
                    outer_prod(0, j, self._nrAncillaDim).full(),
                    self.V_op(j).conj().trans().full(),
                )
            )

        # First Column
        for j in np.arange(1, self._nrOfDampingOps + 1, 1):
            sum = sum + Qobj(
                np.kron(
                    outer_prod(j, 0, self._nrAncillaDim).full(),
                    self.V_op(j).full(),
                )
            )

        return Qobj(
            sum,
            dims=[
                [self._nrAncillaDim, self._systemSizeDim],
                [self._nrAncillaDim, self._systemSizeDim],
            ],
        )

    def H_tilde_second_order(self, dt, t=0):
        """Second order approximation fo H tilde
        https://arxiv.org/pdf/2311.15533 page 27 equation B10

        Args:
                t (float): time stamp
        """

        sum_tmp = np.sqrt(dt) * (self.H_op(t).full()) + np.power(dt, 3 / 2) * (
            -1 / 12 * anti_commute(self.H_op(t).full(), self.sum_of_V_dag_V().full())
        )
        sum = Qobj(np.kron(outer_prod(0, 0, self._nrAncillaDim).full(), sum_tmp))

        for j in np.arange(1, self._nrOfDampingOps + 1, 1):
            sum_tmp = self.V_op(j).full() + dt / 2 * (
                anti_commute(self.V_op(j).full(), self.V_op(0).full())
                + self.V_op_derivative(j).full()
                + 1 / 6 * self.V_op(j).full() @ self.sum_of_V_dag_V().full()
                + 1j / 2 * self.V_op(j).full() @ self.H_op(t).full()
            )
            sum = (
                sum
                + Qobj(np.kron(outer_prod(j, 0, self._nrAncillaDim).full(), sum_tmp))
                + Qobj(
                    np.kron(
                        outer_prod(0, j, self._nrAncillaDim).full(), sum_tmp.conj().T
                    )
                )
            )

        for j in np.arange(1, self._nrOfDampingOps + 1, 1):
            sum_tmp = (
                dt
                / np.sqrt(12)
                * (
                    commute(self.V_op(0).full(), self.V_op(j).full())
                    - self.V_op_derivative(j).full()
                )
            )
            sum = (
                sum
                + Qobj(
                    np.kron(
                        outer_prod(
                            j + self._nrOfDampingOps, 0, self._nrAncillaDim
                        ).full(),
                        sum_tmp,
                    )
                )
                + Qobj(
                    np.kron(
                        outer_prod(
                            0, j + self._nrOfDampingOps, self._nrAncillaDim
                        ).full(),
                        sum_tmp.conj().T,
                    )
                )
            )

        for j in np.arange(1, self._nrOfDampingOps + 1, 1):
            for k in np.arange(1, self._nrOfDampingOps + 1, 1):
                for l in np.arange(1, self._nrOfDampingOps + 1, 1):
                    sum_tmp = (
                        dt
                        / np.sqrt(6)
                        * self.V_op(j).full()
                        @ self.V_op(k).full()
                        @ self.V_op(l).full()
                    )
                    sum = (
                        sum
                        + Qobj(
                            np.kron(
                                outer_prod(
                                    j
                                    + k * self._nrOfDampingOps
                                    + l * self._nrOfDampingOps * self._nrOfDampingOps
                                    - self._nrOfDampingOps * self._nrOfDampingOps
                                    + self._nrOfDampingOps,
                                    0,
                                    self._nrAncillaDim,
                                ).full(),
                                sum_tmp,
                            )
                        )
                        + Qobj(
                            np.kron(
                                outer_prod(
                                    0,
                                    j
                                    + k * self._nrOfDampingOps
                                    + l * self._nrOfDampingOps * self._nrOfDampingOps
                                    - self._nrOfDampingOps * self._nrOfDampingOps
                                    + self._nrOfDampingOps,
                                    self._nrAncillaDim,
                                ).full(),
                                sum_tmp.conj().T,
                            )
                        )
                    )
        for j in np.arange(1, self._nrOfDampingOps + 1, 1):
            for k in np.arange(1, self._nrOfDampingOps + 1, 1):
                sum_tmp = np.sqrt(dt / 2) * self.V_op(j).full() @ self.V_op(k).full()
                sum = (
                    sum
                    + Qobj(
                        np.kron(
                            outer_prod(
                                j
                                + k * self._nrOfDampingOps
                                + self._nrOfDampingOps
                                * self._nrOfDampingOps
                                * self._nrOfDampingOps
                                + self._nrOfDampingOps,
                                0,
                                self._nrAncillaDim,
                            ).full(),
                            sum_tmp,
                        )
                    )
                    + Qobj(
                        np.kron(
                            outer_prod(
                                0,
                                j
                                + k * self._nrOfDampingOps
                                + self._nrOfDampingOps
                                * self._nrOfDampingOps
                                * self._nrOfDampingOps
                                + self._nrOfDampingOps,
                                self._nrAncillaDim,
                            ).full(),
                            sum_tmp.conj().T,
                        )
                    )
                )

        return Qobj(
            sum,
            dims=[
                [self._nrAncillaDim, self._systemSizeDim],
                [self._nrAncillaDim, self._systemSizeDim],
            ],
        )
