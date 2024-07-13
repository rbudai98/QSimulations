import qsimulations
import numpy as np
from qutip import *
import unittest


class Test(unittest.TestCase):
    """
    Unit tests for "qsimulations"
    """

    testObj = qsimulations.qsimulations(0, 0, 0)

    def test_0_set_Hamiltonian(self):
        def H_test(t=0):
            return Qobj(np.array([[0.0, 0.0], [0.0, 1.0]]))

        self.testObj.H_op = H_test
        np.testing.assert_array_equal(
            self.testObj.H_op().full(), np.array([[0.0, 0.0], [0.0, 1.0]])
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
        def H_test(t=0):
            return Qobj(np.array([[0.0, 0.0], [0.0, 1.0]]))

        self.testObj.H_op = H_test
        np.testing.assert_array_equal(
            np.array([[0.0, 0.0], [0.0, 1.0]]), self.testObj.H_op().full()
        )

    def test_5_qobj_hamiltonian_type_test(self):
        def H_test(t=0):
            return Qobj(np.array([[0.0, 0.0], [0.0, 1.0]]))

        self.testObj.H_op = H_test
        self.assertIsInstance(self.testObj.H_op(), Qobj)

    def test_6_test_system_config(self):
        self.testObj.set_system_size(2)
        self.testObj.set_nr_of_ancillas(3)
        self.assertEqual(self.testObj._nrAncillaDim, 8)
        self.assertEqual(self.testObj._systemSizeDim, 4)
        self.assertEqual(self.testObj._totalSystemSizeDim, 32)

    def test_7_Pauli_arrays(self):
        test_I = qsimulations.Pauli_array(qsimulations.I, 1, 1)
        np.testing.assert_array_equal(test_I, np.array([[1.0, 0.0], [0.0, 1.0]]))
        test_X = qsimulations.Pauli_array(qsimulations.X, 1, 1)
        np.testing.assert_array_equal(
            test_X,
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
        )
        test_Y = qsimulations.Pauli_array(qsimulations.Y, 1, 1)
        np.testing.assert_array_equal(test_Y, np.array([[0.0, -1j], [1j, 0.0]]))
        test_Z = qsimulations.Pauli_array(qsimulations.Z, 1, 1)
        np.testing.assert_array_equal(
            test_Z,
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, -1.0],
                ]
            ),
        )

        test_IZ = qsimulations.Pauli_array(qsimulations.Z, 2, 2)
        correctValue = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        )
        np.testing.assert_array_equal(test_IZ, correctValue)

    def test_8_test_damping_operator(self):
        def H_test(t=0):
            return Qobj(
                np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                    ]
                )
            )

        def _test_damping_Fermi_Hubbard(i):
            systemSize = 1
            systemSizeDim = 2
            if i == 0:
                return Qobj(
                    -1j * H_test().full() - 0.5 * np.array([[0.0, 0.0], [1.0, 0.0]])
                )
            if i >= 1 and i <= systemSizeDim:
                return Qobj(
                    0.5
                    * (
                        qsimulations.Pauli_array(qsimulations.X, i, systemSize)
                        + 1j * qsimulations.Pauli_array(qsimulations.Y, i, systemSize)
                    )
                )
            return 0

        self.testObj.V_op = _test_damping_Fermi_Hubbard
        self.testObj.set_system_size(1)
        self.assertEqual(self.testObj._systemSizeDim, 2)
        np.testing.assert_array_equal(
            self.testObj.V_op(1).full(), qsimulations.annihilation_op
        )

    def test_9_outer_prod_test(self):
        self.assertIsInstance(qsimulations.outer_prod(1, 2, 2), Qobj)
        np.testing.assert_array_equal(
            qsimulations.outer_prod(1, 2, 2).full(), np.array([[0.0, 1.0], [0.0, 0.0]])
        )

    def test_10_commute_test(self):
        tmp1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        tmp2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = tmp1 @ tmp2 - tmp2 @ tmp1
        np.testing.assert_array_equal(result, qsimulations.commute(tmp1, tmp2))

    def test_11_anti_commute_test(self):
        tmp1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        tmp2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = tmp1 @ tmp2 + tmp2 @ tmp1
        np.testing.assert_array_equal(result, qsimulations.anti_commute(tmp1, tmp2))

    def test_12_prepare_energy_states(self):
        def H_test(t=0):
            return Qobj(
                np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                    ]
                )
            )

        self.testObj.H_op = H_test
        self.testObj._prep_energy_states()
        np.testing.assert_array_equal(
            np.array([[1.0, 0.0], [0.0, 0.0]]), self.testObj.rho_ground.full()
        )
        np.testing.assert_array_equal(
            np.array([[0.0, 0.0], [0.0, 1.0]]), self.testObj.rho_highest_en.full()
        )
        self.assertIsInstance(self.testObj.rho_ground, Qobj)
        self.assertIsInstance(self.testObj.rho_highest_en, Qobj)

    def test_13_sum_of_V(self):
        def H_test(t=0):
            return Qobj(
                np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                    ]
                )
            )

        def _test_damping_Fermi_Hubbard(i, t=0):
            systemSize = 1
            systemSizeDim = 2
            if i == 0:
                return Qobj(
                    -1j * H_test().full() - 0.5 * np.array([[0.0, 0.0], [1.0, 0.0]])
                )
            if i >= 1 and i <= systemSizeDim:
                return Qobj(
                    0.5
                    * (
                        qsimulations.Pauli_array(qsimulations.X, i, systemSize)
                        + 1j * qsimulations.Pauli_array(qsimulations.Y, i, systemSize)
                    )
                )
            return 0

        self.testObj.V_op = _test_damping_Fermi_Hubbard
        self.testObj.set_nr_of_damping_ops(1)
        self.testObj.set_system_size(1)
        testValue = (
            0.5
            * (qsimulations.X + 1j * qsimulations.Y).conj().T
            @ (0.5 * (qsimulations.X + 1j * qsimulations.Y))
        )
        np.testing.assert_array_equal(self.testObj.sum_of_V_dag_V().full(), testValue)

    def test_14_T_first_ord_type_check(self):
        def H_test(t=0):
            return Qobj(
                np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                    ]
                )
            )

        def _test_damping_Fermi_Hubbard(i, t=0):
            systemSize = 1
            systemSizeDim = 2
            if i == 0:
                return Qobj(
                    -1j * H_test().full() - 0.5 * np.array([[0.0, 0.0], [1.0, 0.0]])
                )
            if i >= 1 and i <= systemSizeDim:
                return Qobj(
                    0.5
                    * (
                        qsimulations.Pauli_array(qsimulations.X, i, systemSize)
                        + 1j * qsimulations.Pauli_array(qsimulations.Y, i, systemSize)
                    )
                )
            return 0

        self.testObj.H_op = H_test
        self.testObj.V_op = _test_damping_Fermi_Hubbard
        self.testObj.set_nr_of_ancillas(2)
        self.testObj.set_nr_of_damping_ops(1)
        self.testObj.set_system_size(1)
        self.assertIsInstance(self.testObj.H_tilde_first_order(0.1), Qobj)

    def test_15_T_second_ord_type_check(self):
        dt = 1

        def H_tmp(t=0):
            return Qobj(np.array([[0.0, 1j], [1.0, 0.0]]))

        self.testObj.H_op = H_tmp

        def V_tmp(i, t=0):
            return Qobj(np.array([[0.0, 2j], [0.0, 0.0]]))

        self.testObj.V_op = V_tmp

        def H_der_tmp(t=0):
            return Qobj(np.array([[0.0, 3j], [0.0, 0.0]]))

        self.testObj.H_op_derivative = H_der_tmp

        def V_der_op(i=0, t=0):
            return Qobj(np.array([[0.0, 4j], [0.0, 0.0]]))

        self.testObj.V_op_derivative = V_der_op

        self.testObj.set_system_size(1)
        self.testObj.set_nr_of_ancillas(3)
        self.testObj.set_nr_of_damping_ops(1)
        self.testObj._update_module_varibles()

        self.assertIsInstance(self.testObj.H_tilde_second_order(dt), Qobj)

    def test_16_check_H_psi_ground_state(self):
        def H_test(t=0):
            return Qobj(
                np.array(
                    [
                        [3.0, 0.0, 0.0, 0.0],
                        [0.0, 4.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0],
                    ]
                )
            )

        self.testObj.H_op = H_test
        self.testObj._prep_energy_states()
        np.testing.assert_array_equal(
            np.array([[0.0, 0.0, 1.0, 0.0]]).astype(complex), self.testObj.psi_ground
        )

    def test_17_check_H_psi_highest_en_state(self):
        def H_test(t=0):
            return Qobj(
                np.array(
                    [
                        [3.0, 0.0, 0.0, 0.0],
                        [0.0, 4.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0],
                    ]
                )
            )

        self.testObj.H_op = H_test
        self.testObj._prep_energy_states()
        np.testing.assert_array_equal(
            np.array([[0.0, 1.0, 0.0, 0.0]]).astype(complex),
            self.testObj.psi_highest_en,
        )

    def test_18_H_first_order_return_value(self):
        def H_test(t=0):
            return Qobj(
                np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                )
            )

        def _damping_op(i, t=0):
            systemSize = 1
            systemSizeDim = 2
            if i == 0:
                return Qobj(
                    -1j * H_test().full() - 0.5 * np.array([[0.0, 0.0], [1.0, 0.0]])
                )
            if i >= 1 and i <= systemSizeDim:
                return Qobj(i * np.array([[0.0, 1j], [-2j, 0.0]]).astype(complex))
            return 0

        self.testObj.H_op = H_test
        self.testObj.V_op = _damping_op
        self.testObj._prep_energy_states()
        dt = 0.1

        testVariable = np.array(
            [
                [1.0 * np.sqrt(dt), 0.0, 0.0, 2j, 0.0, 4j, 0.0, 0.0],
                [0.0, 1.0 * np.sqrt(dt), -1j, 0.0, -2j, 0.0, 0.0, 0.0],
                [0.0, 1j, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-2j, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2j, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-4j, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.testObj.set_nr_of_ancillas(2)
        self.testObj.set_nr_of_damping_ops(2)
        np.testing.assert_array_equal(
            self.testObj.H_tilde_first_order(dt, 0).full(), testVariable
        )

    def test_19_H_second_order_return_value(self):
        dt = 1

        def H_tmp(t=0):
            return Qobj(np.array([[0.0, 1j], [1.0, 0.0]]))

        self.testObj.H_op = H_tmp

        def V_tmp(i=0, t=0):
            if i == 0:
                return Qobj(
                    -1j * H_tmp().full() - 0.5 * np.array([[0.0, 0.0], [0.0, 4.0]])
                )
            else:
                return Qobj(np.array([[0.0, 2j], [0.0, 0.0]]))

        self.testObj.V_op = V_tmp

        def H_der_tmp(t=0):
            return Qobj(np.array([[0.0, 3j], [0.0, 0.0]]))

        self.testObj.H_op_derivative = H_der_tmp

        def V_der_op(i=0, t=0):
            return Qobj(np.array([[0.0, 4j], [0.0, 0.0]]))

        self.testObj.V_op_derivative = V_der_op

        self.testObj.set_system_size(1)
        self.testObj.set_nr_of_ancillas(3)
        self.testObj.set_nr_of_damping_ops(1)
        self.testObj._update_module_varibles()

        testVariable = np.array(
            [
                [
                    0.0,
                    13j / 6,
                    0.5,
                    0.0,
                    -2.0 / np.sqrt(12),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    2 / 3,
                    0.0,
                    -8j / 3,
                    1.0,
                    0.0,
                    2.0 / np.sqrt(12),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.5,
                    8j / 3,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -2.0 / np.sqrt(12),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    2.0 / np.sqrt(12),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ).astype(complex)

        self.assertIsInstance(self.testObj.H_tilde_first_order(0.1), Qobj)
        np.testing.assert_array_almost_equal(
            testVariable, self.testObj.H_tilde_second_order(dt).full()
        )


if __name__ == "__main__":
    unittest.main()
