import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


#Equation de la chaleur 1D

# Paramètres
L = 10.0     # Longueur de la barre
T = 2.0      # Temps total
nx = 100     # Nombre de points en espace
dt = 0.01    # Nombre de points en temps
D = 0.01     # Diffusivité thermique
C = 3   # Coefficient pour le terme advection
dx = L / (nx - 1)   # Discrétisation en espace
nt = int(T / dt)    # Discrétisation en temps
u = np.zeros(nx)    # Initialisation de la température
u[int(nx / 2)] = 1  # Condition initiale: un pic de chaleur au centre
print(C * dt / dx)
print(D * dt / dx**2)
if C * dt / dx > 1:
    print('divergence')
elif D * dt / dx**2 > 1 / 2:
    print('divegence')
else:
    print('convergence')
# Matrice de solution
u_mat = np.zeros((nt, nx))
u_mat[0, :] = u

# Définition de l'exacte solution u_exact
def u_exact(x, t, D=D):
    return np.sin(np.pi * x) * (1 + t) 


x, t = sp.symbols('x t')
u=sp.sin(sp.pi * x) * (1 + t) 
du_dx = sp.diff(u, x)
du_dx_2=sp.diff(du_dx,x)
du_dt = sp.diff(u, t)
f=du_dt-D*du_dx_2+C*du_dx


# Boucle temporelle
for n in range(0, nt):
    u_new = u.copy()
    for i in range(0, nx - 1):
        # Calcul avec ajout de F(x) dans l'équation
        u_new[i] = u[i] + D * dt / dx**2 * (u[i + 1] - 2 * u[i] + u[i - 1]) + C * dt / dx * (u[i + 1] - u[i]) + float(f.subs({x: L, t: n}))*dt # Ajout de la source F(x) ici
    u = u_new.copy()
    u_mat[n, :] = u


# Calcul de l'erreur en norme L2
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
u_exact_mat = np.zeros((nt, nx))
for n in range(nt):
    for i in range(nx):
        u_exact_mat[n, i] = u_exact(x[i], t[n])


# Erreur en norme L2
L2_error = np.sqrt(np.sum((u_mat - u_exact_mat)**2) * dx * dt)
print(f"Erreur en norme L2: {L2_error}")
L2_relative=np.linalg.norm(u_exact_mat)/np.linalg.norm(u_mat)
print(f"Erreur relative L2: {(L2_relative)*100}")

# Affichage
plt.imshow(u_mat, extent=[0, L, T, 0], aspect='auto', cmap='hot')
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Temps t')
plt.title('Évolution de la chaleur en 1D avec source F(x)')
plt.show()
