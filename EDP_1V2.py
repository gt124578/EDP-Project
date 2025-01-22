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

# Solution exacte
def u_exact(x, t):
    return np.sin(np.pi * x) * (1 + t)

# Calcul de l'erreur en fonction du temps
erreurs = []

# Boucle temporelle
for n in range(0, nt):
    u_new = u.copy()
    for i in range(1, nx - 1):
        # Calcul avec ajout de F(x) dans l'équation
        u_new[i] = u[i] + D * dt / dx**2 * (u[i + 1] - 2 * u[i] + u[i - 1]) + C * dt / dx * (u[i + 1] - u[i]) + function_f(i * dx, n*dt) * dt
    u = u_new.copy()
    u_mat[n, :] = u

    # Calcul de l'erreur pour ce temps
    erreur_temp = np.linalg.norm(u - u_exact(np.linspace(0, L, nx), n*dt))  # Norme L2 de l'erreur
    erreurs.append(erreur_temp)

# Affichage de l'erreur en fonction du temps
plt.plot(np.linspace(0, T, nt), erreurs)
plt.xlabel('Temps t')
plt.ylabel('Erreur L2')
plt.title('Erreur en fonction du temps')
plt.grid(True)
plt.show()

# Affichage de la solution numérique et de la solution exacte en fonction du temps à x = L/2
x_fixed = L/2  # Position fixe pour observer la température en fonction du temps
x_index = int(x_fixed / dx)  # L'indice correspondant à x = L/2

# Calcul de la solution exacte à chaque instant de temps
u_exact_values = u_exact(x_fixed, np.linspace(0, T, nt))  # Solution exacte en fonction du temps

# Tracer la solution numérique et exacte en fonction du temps
plt.plot(np.linspace(0, T, nt), u_mat[:, x_index], label='Solution numérique', linestyle='--')
plt.plot(np.linspace(0, T, nt), u_exact_values, label='Solution exacte', linestyle='-')
plt.xlabel('Temps t')
plt.ylabel('Température')
plt.title('Comparaison de la solution numérique et exacte à x = {}'.format(x_fixed))
plt.legend()
plt.grid(True)
plt.show()

# Affichage de la solution dans l'ensemble du domaine
plt.imshow(u_mat, extent=[0, L, T, 0], aspect='auto', cmap='hot')
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Temps t')
plt.title('Évolution de la chaleur en 1D avec source F(x)')
plt.show()
