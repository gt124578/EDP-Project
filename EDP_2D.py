import numpy as np
import matplotlib.pyplot as plt

# Paramètres
Lx, Ly = 1.0, 1.0  # Dimensions du domaine
T = 2.0  # Temps total
nx, ny = 50, 50  # Nombre de points en espace
D = 0.01
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = min(dx, dy)**2 / (4 * D)  # Pas de temps
nt = int(T / dt)  # Discrétisation en temps
Cx=0.03 #Conductivité selon x
Cy=0.01 #Conductivité selon y
# Initialisation des matrices de température
u = np.zeros((nt, nx, ny))
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)
X, Y = np.meshgrid(x, y)

def function_f(x, y, t):
    """Terme source."""
    return np.sin(np.pi * x) * np.sin(np.pi * y) + D * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y) * (1 + t)

# Condition initiale
u[0, :, :] = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Conditions aux limites
def apply_boundary_conditions(u_n):
    """Applique les conditions aux limites Dirichlet."""
    u_n[:, 0] = 0  # Dirichlet à gauche
    u_n[:, -1] = 0  # Dirichlet à droite
    u_n[0, :] = 0  # Dirichlet en bas
    u_n[-1, :] = 0  # Dirichlet en haut
    return u_n

# Boucle temporelle
for n in range(0, nt - 1):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            diffusion = D * ((u[n, i + 1, j] - 2 * u[n, i, j] + u[n, i - 1, j]) / dx**2 +
                             (u[n, i, j + 1] - 2 * u[n, i, j] + u[n, i, j - 1]) / dy**2)
            source = function_f(x[i], y[j], t[n])
            u[n + 1, i, j] = u[n, i, j] + dt * (diffusion + source)
    u[n + 1, :, :] = apply_boundary_conditions(u[n + 1, :, :])

# Calcul de l'erreur en norme L2
def u_exact(x, y, t, D=D):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * (1 + t)

u_exact_mat = np.zeros((nt, nx, ny))
for n in range(nt):
    for i in range(nx):
        for j in range(ny):
            u_exact_mat[n, i, j] = u_exact(x[i], y[j], t[n])

L2_error = np.sqrt(np.sum((u - u_exact_mat)**2) * dx * dy * dt)
print(f"Erreur en norme L2: {L2_error}")

# Affichage des résultats
for n in range(0, nt, nt // 5):
    plt.contourf(X, Y, u[n, :, :], cmap='hot')
    plt.colorbar()
    plt.title(f"Équation de la chaleur en 2D à t={n * dt:.2f}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Tracer l'erreur L2 en fonction du temps
L2_error_time = np.sqrt(np.sum((u - u_exact_mat)**2, axis=(1, 2)) * dx * dy)

plt.plot(t, L2_error_time, label="Erreur L2")
plt.xlabel("Temps")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Erreur L2")
plt.legend()
plt.grid()
plt.show()

# Variation de température en fonction de x pour t donnés
for n in range(0, nt, nt // 5):
    plt.plot(x, u[n, :, ny//2], label=f"t={n * dt:.2f}")

plt.xlabel("Position x")
plt.ylabel("Température u(x, y=L/2, t)")
plt.legend()
plt.title("Variation de la température en fonction de x pour différents t")
plt.show()
