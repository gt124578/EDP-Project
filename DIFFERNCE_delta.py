import numpy as np
import matplotlib.pyplot as plt

# Paramètres
def function_f(x, t, D, C):
    """Terme source."""
    return np.sin(np.pi * x) + D * (np.pi**2) * np.sin(np.pi * x) * (1 + t) + C * np.pi * np.cos(np.pi * x) * (1 + t)

def u_exact(x, t, D):
    return np.sin(np.pi * x) * (1 + t)

def apply_boundary_conditions(u_n):
    """Applique les conditions aux limites Dirichlet."""
    u_n[0] = 0  # Dirichlet à gauche
    u_n[-1] = 0  # Dirichlet à droite
    return u_n

# Différentes valeurs de nx
nx_values = [50, 100, 200, 400]  # Résolutions spatiales
L = 1.0     # Longueur de la barre
T = 2.0     # Temps total
D = 0.01    # Coefficient de diffusion
C = 0.03    # Coefficient de convection

def solve_heat_eq(nx, L, T, D, C):
    dx = L / (nx - 1)
    dt = dx**2 / (2 * D)  # Pas de temps selon la condition CFL
    nt = int(T / dt)

    x = np.linspace(0, L, nx)
    u = np.zeros((nt, nx))  # Initialisation de la température
    u[0, :] = np.sin(np.pi * x)  # Condition initiale

    for n in range(nt - 1):
        for i in range(1, nx - 1):
            convection = -C * (u[n, i + 1] - u[n, i - 1]) / (2 * dx)
            diffusion = D * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / dx**2
            source = function_f(i * dx, n * dt, D, C)
            u[n + 1, i] = u[n, i] + dt * (convection + diffusion + source)
        u[n + 1, :] = apply_boundary_conditions(u[n + 1, :])

    t = np.linspace(0, T, nt)
    u_exact_mat = np.array([[u_exact(x[i], t[n], D) for i in range(nx)] for n in range(nt)])
    L2_error = np.sqrt(np.sum((u - u_exact_mat)**2, axis=1) * dx)  # Erreur en norme L2 pour chaque instant
    return x, t, L2_error

# Résolution pour chaque nx et tracé des erreurs
plt.figure()
for nx in nx_values:
    x, t, L2_error = solve_heat_eq(nx, L, T, D, C)
    plt.plot(t, L2_error, label=f"dx = {L / (nx - 1):.4f}")

plt.xlabel("Temps t")
plt.ylabel("Erreur L2")
plt.title("Évolution de l'erreur L2 pour différents dx")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
