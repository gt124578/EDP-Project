# Numerical Simulation of the 1D & 2D Advection-Diffusion Equation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)

This project presents a Python-based numerical solver for the 1D and 2D advection-diffusion equation, a fundamental Partial Differential Equation (PDE) used to model a wide variety of physical phenomena, such as heat transfer and particle transport.

The primary objective was to implement a robust Finite Difference Method (FDM) scheme, analyze its accuracy against a known exact solution, and study its convergence properties with respect to spatial and temporal discretization steps.

## Key Features

- **1D and 2D Solvers:** Implements a Forward-Time Centered-Space (FTCS) finite difference scheme for both 1D and 2D domains.
- **Advection & Diffusion:** The model correctly simulates both diffusion (heat spreading) and convection (heat being carried by a flow) terms.
- **Symbolic Source Term:** Utilizes the `SymPy` library to symbolically derive the required source term `f(x, t)` from a chosen exact solution, allowing for rigorous error analysis.
- **L2 Norm Error Analysis:** Quantitatively measures the accuracy of the numerical solution by calculating the L2 norm of the error against the exact solution over time.
- **Convergence Study:** Systematically analyzes the evolution of the L2 error for different spatial (`dx`) and temporal (`dt`) step sizes to validate the theoretical convergence rate of the numerical scheme.

## Results & Visualizations

The solver successfully simulates the evolution of the temperature field and its error metrics provide insight into the method's performance.

### 1D Simulation: Temperature Evolution

The animation below shows the evolution of the temperature profile `u(x, t)` along a 1D rod over time. The solution correctly adheres to the initial sinusoidal profile and the zero-value Dirichlet boundary conditions.

![1D Evolution](results/1d_heat_evolution.png)

### 2D Simulation: Heatmap

This heatmap illustrates the temperature distribution `u(x, y, t)` on a 2D plate at a specific time `t`. The simulation captures the combined effects of diffusion spreading the heat and convection shifting the peak temperature.

![2D Heatmap](results/2d_heatmap.png)

### Convergence Analysis

The plot below demonstrates the convergence of the numerical scheme. As the spatial discretization `dx` is refined (made smaller), the L2 error decreases, confirming the consistency and accuracy of the implementation.

![Error vs dx](results/1d_error_vs_dx.png)

## Methodology

The project solves the advection-diffusion equation:

$$ \frac{\partial u}{\partial t} = D \nabla^2 u - \vec{C} \cdot \nabla u + f(\vec{x}, t) $$

Where:
- `u` is the temperature field.
- `D` is the diffusion coefficient.
- `\vec{C}` is the convection velocity vector.
- `f` is the source term.

The numerical solution is obtained using a **Forward-Time Centered-Space (FTCS)** finite difference scheme. The simulation is stabilized by respecting the Courant-Friedrichs-Lewy (CFL) condition, which links the time step `dt` to the spatial step `dx`.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── EDP_Project_1D_and_2D.py         # Main script to run simulations and generate plots
├── results/                # Directory for saved plot images
    ├── 1D_Heat_Equation.png
    ├── 1d_error_vs_dt.png
    ├── 1d_error_vs_dx.png
    ├── 1d_heat_evolution.png
    ├── 2D_Heat_Equation.png
    ├── 2d_error_vs_exact_solution.png
    ├── 2d_heat_evolution_in_t.png
    ├── output.txt          # Output in the console (L2 error in 1D and 2D)
```

## How to Run


## Dependencies

- NumPy
- Matplotlib
- SymPy
