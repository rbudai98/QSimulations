# QSimulations [![Python Package using Conda](https://github.com/rbudai98/QSimulations/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/rbudai98/QSimulations/actions/workflows/python-package-conda.yml)
Python library for quantum systems and simulation

## Overview
This module provides various tools and utilities for simulating quantum systems. It includes functions for handling common quantum operations, such as Pauli operators, outer products, commutators, and anti-commutators. Additionally, it provides a class `qsimulations` to manage and simulate time-dependent quantum systems with ancillary qubits and damping operators.

## Articles
* [Single-ancilla ground state preparation via Lindbladians](https://arxiv.org/abs/2308.15676)
* [Simulating Open Quantum Systems Using Hamiltonian Simulations](https://arxiv.org/abs/2311.15533)

## Simulations:
* `TFIM_time_indep_multi_qubit_.ipynb`: Time independent transverse-field Ising model
* `TFIM_time_dependent_mult_qub.ipynb`: Time dependent transverse-field Ising model
* `Fermi_Hub_multi_qubit.ipynb`: Fermi-Hubbard model

## Functions

### Utility Functions

#### Pauli_array
```python
def Pauli_array(op, poz, size):
```
Generates a Pauli operator for a given position in a system.

- `op` (np.array): The Pauli operator (X, Y, Z).
- `poz` (int): Position of the operator in the system.
- `size` (int): Total size of the system.

#### outer_prod
```python
def outer_prod(left_poz, right_poz, size):
```
Computes the outer product of two states.

- `left_poz` (int): Position of the 1 in the ket vector.
- `right_poz` (int): Position of the 1 in the bra vector.
- `size` (int): Size of the overall system.

#### commute
```python
def commute(op1, op2):
```
Calculates the commutator of two operators.

- `op1` (np.array): First operator.
- `op2` (np.array): Second operator.

#### anti_commute
```python
def anti_commute(op1, op2):
```
Calculates the anti-commutator of two operators.

- `op1` (np.array): First operator.
- `op2` (np.array): Second operator.

#### Taylor_approximation
```python
def Taylor_approximtion(H_op, order, dt, initial_state):
```
Computes the Taylor approximation for a given Hamiltonian.

- `H_op` (Qobj): Hamiltonian to be approximated.
- `order` (int): Order of the approximation.
- `dt` (float): Time step.
- `initial_state` (Qobj): Initial state of the system.

## Class: `qsimulations`
The `qsimulations` class provides a framework for setting up and running quantum simulations with ancillary qubits and damping operators.

### Initialization
```python
def __init__(self, H=0, systemSize=0, nrOfDampingOps=0, nrOFAncillas=0):
```
Initializes the simulation parameters.

- `H` (Qobj): Hamiltonian of the system.
- `systemSize` (int): Size of the quantum system.
- `nrOfDampingOps` (int): Number of damping operators.
- `nrOFAncillas` (int): Number of ancillary qubits.

### Methods

#### `set_H_op`
```python
def set_H_op(self, H_tmp):
```
Sets the Hamiltonian of the system.

- `H_tmp` (Qobj): New Hamiltonian.

#### `set_system_size`
```python
def set_system_size(self, size):
```
Sets the size of the quantum system.

- `size` (int): New system size.

#### `set_nr_of_damping_ops`
```python
def set_nr_of_damping_ops(self, nr):
```
Sets the number of damping operators.

- `nr` (int): Number of damping operators.

#### `set_nr_of_ancillas`
```python
def set_nr_of_ancillas(self, nr):
```
Sets the number of ancillary qubits.

- `nr` (int): Number of ancillary qubits.

#### `H_op`
```python
def H_op(self):
```
Returns the Hamiltonian of the system.

#### `V_op`
```python
def V_op(self, i):
```
Returns the damping operator for a given index.

- `i` (int): Index of the damping operator.

#### `H_op_derivative`
```python
def H_op_derivative(self):
```
Returns the time derivative of the Hamiltonian.

#### `V_op_derivative`
```python
def V_op_derivative(self, i):
```
Returns the time derivative of a damping operator.

- `i` (int): Index of the damping operator.

#### `sum_of_V_dag_V`
```python
def sum_of_V_dag_V(self):
```
Returns the summation of all damping operators.

#### `H_tilde_first_order`
```python
def H_tilde_first_order(self, dt):
```
Calculates the first-order approximation of the Hamiltonian.

- `dt` (float): Time step.

#### `H_tilde_second_order`
```python
def H_tilde_second_order(self, dt):
```
Calculates the second-order approximation of the Hamiltonian.

- `dt` (float): Time step.