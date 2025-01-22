import numpy as np
import matplotlib.pyplot as plt

# Paramètres
L = 10.0     # Longueur de la barre
T = 2.0      # Temps total
nx = 100     # Nombre de points en espace
dt = 0.01    # Pas de temps
D = 0.01     # Diffusivité thermique
C = 3        # Coefficient pour le terme advection
dx = L / (nx - 1)   # Discrétisation en espace
nt = int(T / dt)    # Discrétisation en temps
u = np.zeros(nx)    # Initialisation de la température
u[0] = 1  # Condition initiale: un pic de chaleur au centre

# Conditions de stabilité
print(C * dt / dx)
print(D * dt / dx**2)
if C * dt / dx > 1:
    print('Divergence due à l\'advection')
elif D * dt / dx**2 > 1 / 2:
    print('Divergence due à la diffusion')
else:
    print('Convergence')

# Matrice de solution
u_mat = np.zeros((nt, nx))
u_mat[0, :] = u

def function_f(x, t):
    return np.sin(np.pi * x) + D * np.pi**2 * np.sin(np.pi * x) * (1 + t) + C * np.pi * np.cos(np.pi * x) * (1 + t)

# Boucle temporelle
for n in range(0, nt):
    u_new = u.copy()
    for i in range(1, nx - 1):
        # Calcul avec ajout de F(x) dans l'équation
        u_new[i] = u[i] + D * dt / dx**2 * (u[i + 1] - 2 * u[i] + u[i - 1]) + C * dt / dx * (u[i + 1] - u[i]) + function_f(i * dx, n*dt) * dt
    u = u_new.copy()
    u_mat[n, :] = u

# Affichage
plt.imshow(u_mat, extent=[0, L, T, 0], aspect='auto', cmap='hot')
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Temps t')
plt.title('Évolution de la chaleur en 1D avec source F(x)')
plt.show()

# Calcul de la solution exacte
def u_exact(x, t):
    return np.sin(np.pi * x) * (1 + t)

# Calcul de l'erreur relative
erreur_absolute = np.zeros(nx)
norme_u_exacte = np.linalg.norm(u_exact(np.linspace(0, L, nx), T))  # Norme L2 de la solution exacte au temps final
for i in range(nx):
    erreur_absolute[i] = u[i] - u_exact(i * dx, T)  # Erreur à chaque point spatial

erreur_L2 = np.linalg.norm(erreur_absolute)  # Norme L2 de l'erreur
erreur_relative = erreur_L2 / norme_u_exacte  # Erreur relative

print("Erreur relative : ", erreur_relative)
