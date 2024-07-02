from scipy.linalg import sqrtm, cosm, sinm, expm
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.linalg as la
import qsimulations
import numpy as np
from qutip import *
import unittest
import math
import qib


class Test(unittest.TestCase):
    """
    Unit tests for "qsimulations"
    """

    testObj = qsimulations.qsimulations(0, 0, 0, 0)

    def test_0_set_Hamiltonian(self):
        self.testObj.set_H_op(np.matrix([[0.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_array_equal(
            self.testObj._H, np.matrix([[0.0, 0.0], [0.0, 1.0]])
        )

    def test_1_set_system_size(self):
        self.testObj.set_system_size(1)
        self.assertEqual(self.testObj._systemSize, 1)

    def test_2_set_nr_damping_ops(self):
        self.testObj.set_nr_of_damping_ops(1)
        self.assertEqual(self.testObj._nrOfDampingOps, 1)

    def test_3_set_nr_of_ancillas(self):
        self.testObj.set_nr_of_ancillas(1)
        self.assertEqual(self.testObj._nrAncilla, 1)

    def test_4_qobj_hamiltonian_test(self):
        self.testObj.set_H_op(np.matrix([[0.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_array_equal(
            np.matrix([[0.0, 0.0], [0.0, 1.0]]), self.testObj.H_op().full()
        )

    def test_5_qobj_hamiltonian_type_test(self):
        self.testObj.set_H_op(np.matrix([[0.0, 0.0], [0.0, 1.0]]))
        self.assertIsInstance(self.testObj.H_op(), Qobj)

    def test_6_test_system_config(self):
        self.testObj.set_system_size(2)
        self.testObj.set_nr_of_ancillas(3)
        self.assertEquals(self.testObj._nrAncillaDim, 8)
        self.assertEquals(self.testObj._systemSizeDim, 4)
        self.assertEquals(self.testObj._totalSystemSizeDim, 32)

    def test_7_Pauli_arrays(self):
        test_I = self.testObj.Pauli_array(self.testObj.I, 1, 1)
        np.testing.assert_array_equal(test_I, np.array([[1.0, 0.0], [0.0, 1.0]]))
        test_X = self.testObj.Pauli_array(self.testObj.X, 1, 1)
        np.testing.assert_array_equal(
            test_X,
            np.matrix(
                [
                    [
                        0.0,
                        1.0,
                    ],
                    [1.0, 0.0],
                ]
            ),
        )
        test_Y = self.testObj.Pauli_array(self.testObj.Y, 1, 1)
        np.testing.assert_array_equal(test_Y, np.matrix([[0.0, -1j], [1j, 0.0]]))
        test_Z = self.testObj.Pauli_array(self.testObj.Z, 1, 1)
        np.testing.assert_array_equal(
            test_Z,
            np.matrix(
                [
                    [
                        1.0,
                        0.0,
                    ],
                    [0.0, -1.0],
                ]
            ),
        )

        test_IZ = self.testObj.Pauli_array(self.testObj.Z, 2, 2)
        correctValue = np.matrix(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        )
        np.testing.assert_array_equal(test_IZ, correctValue)

    def test_8_test_damping_operator(self):
        self.testObj.V_op = _test_damping_Fermi_Hubbard
        self.testObj.set_system_size(1)
        self.assertEqual(self.testObj._systemSizeDim, 2)
        np.testing.assert_array_equal(
            self.testObj.V_op(self.testObj, 1).full(), self.testObj.annihilation_op
        )

    def test_9_outer_prod_test(self):
        self.assertIsInstance(self.testObj.outer_prod(1, 2, 2), Qobj)
        np.testing.assert_array_equal(
            self.testObj.outer_prod(1, 2, 2).full(), np.matrix([[0.0, 1.0], [0.0, 0.0]])
        )

    def test_10_commute_test(self):
        tmp1 = np.matrix([[1.0, 2.0], [3.0, 4.0]])
        tmp2 = np.matrix([[5.0, 6.0], [7.0, 8.0]])
        result = tmp1 @ tmp2 - tmp2 @ tmp1
        np.testing.assert_array_equal(result, self.testObj.commute(tmp1, tmp2))

    def test_11_anti_commute_test(self):
        tmp1 = np.matrix([[1.0, 2.0], [3.0, 4.0]])
        tmp2 = np.matrix([[5.0, 6.0], [7.0, 8.0]])
        result = tmp1 @ tmp2 + tmp2 @ tmp1
        np.testing.assert_array_equal(result, self.testObj.anti_commute(tmp1, tmp2))


def _test_damping_Fermi_Hubbard(self, i):
    if i == 0:
        return Qobj(-1j * self.H_op() - 0.5 * self.sum_of_V_dag_V(0))
    if i >= 1 and i <= self._systemSizeDim:
        return Qobj(
            0.5
            * (
                self.Pauli_array(self.X, i, self._systemSize)
                + 1j * self.Pauli_array(self.Y, i, self._systemSize)
            )
        )
    return 0


if __name__ == "__main__":
    unittest.main()
