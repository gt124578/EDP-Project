import numpy as np
import matplotlib.pyplot as plt

# Paramètres
L = 10.0     # Longueur de la barre
T = 2.0      # Temps total
nx = 100     # Nombre de points en espace
nt = 1000    # Nombre de points en temps

alpha = 0.01 # Diffusivité thermique
kappa = 0.5  # Conductivité thermique
rho = 1.0    # Densité
c = 1.0      # Capacité thermique

dx = L / (nx - 1)   # Discrétisation en espace
dt = T / nt         # Discrétisation en temps

# Initialisation de la température
u = np.zeros(nx)    
u[int(nx/2)] = 1    # Condition initiale: un pic de chaleur au centre

# Matrice de solution
u_mat = np.zeros((nt, nx))
u_mat[0, :] = u

# Boucle temporelle
for n in range(1, nt):
    u_new = u.copy()
    for i in range(1, nx-1):
        u_new[i] = u[i] + (alpha + kappa/(rho*c)) * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new.copy()
    u_mat[n, :] = u

# Affichage
plt.imshow(u_mat, extent=[0, L, T, 0], aspect='auto', cmap='hot')
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Temps t')
plt.title('Évolution de la chaleur en 1D avec conductivité')
plt.show()