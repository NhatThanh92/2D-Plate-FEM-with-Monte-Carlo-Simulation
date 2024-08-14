# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:07:34 2024

@author: thanh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from Solution import solution
from Shapefunc import shapeFuncQ4
from Jacob import Jacobian
from GaussQuad import gaussQuadrature
from Plotmesh import PlotMesh
import time  # Import time module for measuring computation time
# Material Properties
E_mean = 2e11  # Mean Young's modulus
E_std_dev = 2e10  # Standard deviation of Young's modulus
h = 0.01
pois = 0.3
Lx = 1
Ly = 1
tol = 1e-6

# Stress-strain matrix (C): Plane stress condition
E1 = E_mean / (1 - pois**2)
C = E1 * np.array([[1, pois, 0],
                   [pois, 1, 0],
                   [0, 0, (1 - pois) / 2]])

# Function to compute stiffness matrix
def formStiffness2D(nDof, nE, eNodes, nP, xy, C, h):
    K = np.zeros((nDof, nDof))
    gaussWt, gaussLoc = gaussQuadrature(1)
    for e in range(nE):
        id = eNodes[e, :]
        eDof = np.zeros((8, 1))
        eDof[0:4, 0] = id
        eDof[4:8, 0] = id + nP
        eDof = eDof.flatten()

        ndof = id.size
        for q in range(gaussWt.size):
            GaussPoint = gaussLoc[q, :]
            xi = GaussPoint[0]
            eta = GaussPoint[1]

            shape, nDeriv = shapeFuncQ4(xi, eta)
            J, xyDeriv = Jacobian(xy[id - 1, :], nDeriv)

            B = np.zeros((3, 2 * ndof))
            B[0, 0:ndof] = np.transpose(xyDeriv[:, 0])
            B[1, ndof:(2 * ndof)] = np.transpose(xyDeriv[:, 1])
            B[2, 0:ndof] = np.transpose(xyDeriv[:, 1])
            B[2, ndof:(2 * ndof)] = np.transpose(xyDeriv[:, 0])

            BT = np.transpose(B)
            detJ = np.linalg.det(J)
            Ke = np.matmul(np.matmul(BT, C), B) * h * detJ * gaussWt[q]

            for ii in range(np.size(Ke, 0)):
                row = int(eDof[ii]) - 1
                for jj in range(np.size(Ke, 1)):
                    col = int(eDof[jj]) - 1
                    K[row, col] += Ke[ii, jj]
    return K

# Import nodal coordinates
dfnode = pd.read_csv('nodal_coordinates.csv', header=None).dropna(axis=0)
nP = dfnode.shape[0]
nDof = 2 * nP

xy = dfnode.iloc[:, 1:3].to_numpy(dtype=np.float32)

# Import nodal connectivities
dfeNode = pd.read_csv('nodal_connectivities.csv', header=None).dropna(axis=0)
eNodes = dfeNode.iloc[:, 6:10].astype(int).to_numpy(dtype=np.int32)
nE = eNodes.shape[0]

# Calculate stiffness matrix
K = formStiffness2D(nDof, nE, eNodes, nP, xy, C, h)

# Boundary condition
fixP = np.argwhere(xy[:, 0] <= tol)
fixDof = np.hstack([fixP, fixP + nP]).flatten()

# Loading
dfx = 1e8
dy = 1 / 20

force = np.zeros((nDof, 1))
loadP1 = np.where((xy[:, 0] >= Lx - tol) & (xy[:, 1] >= Ly - tol))
loadP2 = np.where((xy[:, 0] >= Lx - tol) & (xy[:, 1] <= tol))
loadP3 = np.where((xy[:, 0] >= Lx - tol) & (xy[:, 1] >= tol) & (xy[:, 1] <= Ly - tol))
force[loadP1, 0] = dfx * dy / 2
force[loadP2, 0] = dfx * dy / 2
force[loadP3, 0] = dfx * dy

# Number of Monte Carlo simulations
num_simulations = 100

# Arrays to store displacements
displacements = np.zeros((num_simulations, nDof))
# Measure computation time for Monte Carlo simulations
start_time = time.time()
# Monte Carlo simulation loop
for i in range(num_simulations):
    # Generate sample of Young's modulus
    E_sample = np.random.normal(E_mean, E_std_dev)
    E1_sample = E_sample / (1 - pois**2)
    C_sample = E1_sample * np.array([[1, pois, 0],
                                     [pois, 1, 0],
                                     [0, 0, (1 - pois) / 2]])

    # Calculate stiffness matrix
    K_sample = formStiffness2D(nDof, nE, eNodes, nP, xy, C_sample, h)

    # Solve for displacements
    disp_sample = solution(nDof, fixDof, K_sample, force)

    # Store displacements
    displacements[i, :] = disp_sample.flatten()

# Compute mean and standard deviation of displacements
mean_disp = np.mean(displacements, axis=0)
std_disp = np.std(displacements, axis=0)
# Print total computation time for Monte Carlo simulations
total_time = time.time() - start_time
print(f"Total computation time: {total_time:.4f} seconds")

# Plot CDF of displacements
plt.figure(figsize=(10, 6))
sorted_disp = np.sort(displacements.flatten())
cdf = np.arange(len(sorted_disp)) / float(len(sorted_disp))
plt.plot(sorted_disp, cdf, label='CDF')

plt.xlabel('Displacement')
plt.ylabel('Cumulative Probability')
plt.title(f'CDF of Displacement with Uncertainty in Young\'s Modulus\n'
          f'Computation Time: {total_time:.4f} seconds\n'
          f'Number of Simulations: {num_simulations}')
plt.grid(True)
plt.legend()

# Plot PDF of displacements
plt.figure(figsize=(10, 6))
plt.hist(displacements.flatten(), bins=20, density=True, alpha=0.75, label='PDF')
plt.xlabel('Displacement')
plt.ylabel('Probability Density')
plt.title(f'PDF of Displacement with Uncertainty in Young\'s Modulus\n'
          f'Computation Time: {total_time:.4f} seconds\n'
          f'Number of Simulations: {num_simulations}')
plt.grid(True)
plt.legend()

# Show plots
plt.show()