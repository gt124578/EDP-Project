import numpy as np
import matplotlib.pyplot as plt
import sympy as sp



# Fixer nx et varier dt
nx = 100  # Résolution spatiale fixe
L = 1.0   # Longueur de la barre
T = 2.0   # Temps total
D = 0.01  # Coefficient de diffusion
C = 0.03  # Coefficient de convection
dt_factors = [0.1, 0.5, 1.0]  # Facteurs de variation pour dt



#Permet de calculer f(x,t) pour u quelconque par le calcul formel, il suffit de modifier u_ex dans la fonction
def determinate_f():
    y, u = sp.symbols('y u') #les variables sont y et t pour la résolution formelle 
    #u_ex=sp.sin(np.pi*y)*(1+t)
    u_ex = (1/np.sqrt(2*np.pi))*sp.exp(-(y**2)/2)*(1+u) #gaussienne avec mu=0 et sigma=1
    du_dx = sp.diff(u_ex, y)
    du_dx_2 = sp.diff(du_dx, y)
    du_dt = sp.diff(u_ex, u)
    f = du_dt - D * du_dx_2 + C * du_dx
    print(f)
    return f

y, u = sp.symbols('y u')
f=determinate_f()

def function_f(x, tps, know_f=True, f=f, D=D, C=C):
    if know_f:
        """Terme source."""
        return np.sin(np.pi * x) + D * (np.pi**2) * np.sin(np.pi * x) * (1 + tps) + C * np.pi * np.cos(np.pi * x) * (1 + tps)
    else:
        return float(f.subs({y: x, u: tps}))


def u_exact(x, t):
    return np.sin(np.pi*x)*(1+t)

def apply_boundary_conditions(u_n):
    """Applique les conditions aux limites Dirichlet."""
    u_n[0] = 0  # Dirichlet à gauche
    u_n[-1] = 0  # Dirichlet à droite
    return u_n


def solve_heat_eq_with_dt(nx, L, T, D, C, dt_factor):
    dx = L / (nx - 1)
    dt = dt_factor * (dx**2 / (2 * D))  # Variation de dt avec un facteur
    nt = int(T / dt)

    x = np.linspace(0, L, nx)
    u = np.zeros((nt, nx))  # Initialisation de la température
    u[0, :] = np.sin(np.pi * x)  # Condition initiale

    for n in range(nt - 1):
        for i in range(1, nx - 1):
            convection = -C * (u[n, i + 1] - u[n, i - 1]) / (2 * dx)
            diffusion = D * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / dx**2
            source = function_f(i * dx, n * dt)
            u[n + 1, i] = u[n, i] + dt * (convection + diffusion + source)
        u[n + 1, :] = apply_boundary_conditions(u[n + 1, :])

    t = np.linspace(0, T, nt)
    u_exact_mat = np.array([[u_exact(x[i], t[n]) for i in range(nx)] for n in range(nt)])
    L2_error = np.sqrt(np.sum((u - u_exact_mat)**2, axis=1) * dx)   # Erreur en norme L2 pour chaque instant
    return t, L2_error

# Résolution pour chaque dt et tracé des erreurs
plt.figure()
for dt_factor in dt_factors:
    t, L2_error = solve_heat_eq_with_dt(nx, L, T, D, C, dt_factor)
    plt.plot(t, L2_error, label=f"dt = {dt_factor:.2f} * CFL")

plt.xlabel("Temps t")
plt.ylabel("Erreur L2")
plt.title("Évolution de l'erreur L2 pour différents dt")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
