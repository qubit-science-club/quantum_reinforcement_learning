import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable

np.random.seed(42)

# we generate a three-dimensional random vector by sampling
# each entry from a standard normal distribution
v = np.random.normal(0, 1, 3)

# purity of the target state
purity = 0.66

# create a random Bloch vector with the specified purity
bloch_v = np.sqrt(2 * purity - 1) * v / np.sqrt(np.sum(v ** 2))
# bloch_v = np.array([0.0, 0.0, 1.0])

# array of Pauli matrices (will be useful later)
Paulis = np.zeros((3, 2, 2), dtype=complex)
Paulis[0] = [[0, 1], [1, 0]]
Paulis[1] = [[0, -1j], [1j, 0]]
Paulis[2] = [[1, 0], [0, -1]]

# number of qubits in the circuit
nr_qubits = 3
# number of layers in the circuit
nr_layers = 2

# randomly initialize parameters from a normal distribution. Below
# - first parameter (equal to 0) is the “centre” of the distribution,
# - second parameter (equal to pi) is the standard deviation of the distribution
params = np.random.normal(0, np.pi, (nr_qubits, nr_layers, 3))
# params = np.zeros((nr_qubits, nr_layers, 3))
params = Variable(torch.tensor(params), requires_grad=True)

# a layer of the circuit ansatz
def layer(params, j):
    for i in range(nr_qubits):
        qml.RX(params[i, j, 0], wires=i)
        qml.RY(params[i, j, 1], wires=i)
        qml.RZ(params[i, j, 2], wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 2])

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev, interface="torch")
def circuit(params, A=None):

    # repeatedly apply each layer in the circuit
    for j in range(nr_layers):
        layer(params, j)

    # returns the expectation of the input matrix A on the first qubit
    return qml.expval(qml.Hermitian(A, wires=0))

# cost function
def cost_fn(params):
    cost = 0
    for k in range(3):
        cost += torch.abs(circuit(params, A=Paulis[k]) - bloch_v[k])

    return cost

# set up the optimizer
opt = torch.optim.Adam([params], lr=0.1)

# number of steps in the optimization routine
steps = 200

# the final stage of optimization isn't always the best, so we keep track of
# the best parameters along the way
best_cost = cost_fn(params)
best_params = np.zeros((nr_qubits, nr_layers, 3))

print("Cost after 0 steps is {:.4f}".format(cost_fn(params)))

# optimization begins
for n in range(steps):
    opt.zero_grad()
    loss = cost_fn(params)
    loss.backward()
    opt.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_params = params

    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Cost after {} steps is {:.4f}".format(n + 1, loss))

# calculate the Bloch vector of the output state
output_bloch_v = np.zeros(3)
for l in range(3):
    output_bloch_v[l] = circuit(best_params, A=Paulis[l])

print("Target Bloch vector = ", bloch_v)
print("Output Bloch vector = ", output_bloch_v)
print(circuit.draw())

# This code would give such an output:
# Cost after 0 steps is 1.0179
# Cost after 10 steps is 0.1467
# Cost after 20 steps is 0.0768
# ...
# Cost after 190 steps is 0.0502
# Cost after 200 steps is 0.0573
# Target Bloch vector =  [ 0.33941241 -0.09447812  0.44257553]
# Output Bloch vector =  [ 0.3070773  -0.07421859  0.47392787]
# Found circuit:
#  0: ──RX(4.974)───RY(-0.739)──RZ(-0.358)──╭C──╭C───RX(4.6)──RY(2.739)───RZ(-1.297)──────────────╭C──╭C──────┤ ⟨H0⟩
#  1: ──RX(1.927)───RY(-1.859)──RZ(-1.008)──╰X──│───╭C────────RX(0.375)───RY(-6.204)──RZ(-5.583)──╰X──│───╭C──┤
#  2: ──RX(-2.027)──RY(-3.447)──RZ(1.425)───────╰X──╰X────────RX(-2.378)──RY(-4.139)──RZ(4.284)───────╰X──╰X──┤
# H0 =
# [[ 1.+0.j  0.+0.j]
#  [ 0.+0.j -1.+0.j]]
