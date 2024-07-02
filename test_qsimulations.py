from scipy.linalg import sqrtm, cosm, sinm, expm
import matplotlib.pyplot as plt
from scipy.special import erf
import scipy.linalg as la
import qusimulations
import numpy as np
from qutip import *
import unittest
import math
import qib


class Test(unittest.TestCase):
    """
    Unit tests for "lib_robi"
    """

    testObj = qusimulations.qsimulations(0, 0, 0, 0)

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
        np.testing.assert_array_equal(test_I, self.testObj.I)
        test_X = self.testObj.Pauli_array(self.testObj.X, 1, 1)
        np.testing.assert_array_equal(test_X, self.testObj.X)
        test_Y = self.testObj.Pauli_array(self.testObj.Y, 1, 1)
        np.testing.assert_array_equal(test_Y, self.testObj.Y)
        test_Z = self.testObj.Pauli_array(self.testObj.Z, 1, 1)
        np.testing.assert_array_equal(test_Z, self.testObj.Z)

        test_IZ = self.testObj.Pauli_array(self.testObj.Z, 2, 2)
        correctValue = np.matrix(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        )
        np.testing.assert_array_equal(test_IZ, correctValue)


if __name__ == "__main__":
    unittest.main()
