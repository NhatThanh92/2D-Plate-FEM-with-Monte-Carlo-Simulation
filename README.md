# 2D-Plate-FEM-with-Monte-Carlo-Simulation
## Introduction
In many structural engineering problems, the material properties like Young's modulus can vary due to manufacturing processes or material inconsistencies. This project uses Monte Carlo simulation to quantify the uncertainty in the displacement of a 2D structure caused by variations in the Young's modulus. The displacement response of the structure is studied using a finite element analysis under plane stress conditions.

## Features
Finite Element Analysis (FEA): Performs 2D plane stress analysis using the FEM.

Monte Carlo Simulation: Quantifies the uncertainty in structural displacements due to random variations in Young's modulus.

Customizable Simulations: Easily adjust the number of Monte Carlo simulations and material properties.

Visualization: Provides probability density function (PDF) and cumulative distribution function (CDF) plots for the displacement, highlighting the probabilistic nature of the structural response.

## Results


![image](https://github.com/user-attachments/assets/24a69898-a959-4dd3-8a4b-9234b8a8d9bb)

**Fig 1. Contour Plots of Displacement Fields U and V.**


![image](https://github.com/user-attachments/assets/c9551430-8bb7-4a95-9f9e-618da29e53ec)

**Fig 2. CDF and PDF of displacement under 100 realizations.** 

![100_1000_5000](https://github.com/user-attachments/assets/2c556e9f-8998-460a-8dc5-5771727ad757)

**Fig 3. CDF of Displacement for 100, 1000, and 5000 Simulations.**
