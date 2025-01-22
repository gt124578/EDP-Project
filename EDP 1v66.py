import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Paramètres 
L = 1     # Longueur de la barre
nx = 100     # Nombre de points en espace
D = 0.01     # Diffusivité thermique
T = 2.0      # Temps total
dt = 0.01    # Nombre de points en temps
x = np.linspace(0, L, nx)
t_fixed = 1.0  # Temps fixé
nt = int(T / dt)    # Discrétisation en temps
dx = L / (nx - 1)   # Discrétisation en espace
C = 0.03   # Coefficient pour le terme advection
u = np.zeros(nx)    # Initialisation de la température
# Matrice de solution
u_mat = np.zeros((nt, nx))
u_mat[0, :] = u





# Définition de l'exacte solution u_exact
def u_exact(x, t, D=D):
    return np.sin(np.pi * x)*(1 + t) 

for t_fixed in range(0,nt):
    if t_fixed%10==0:
        # Affichage
        plt.plot(x, u_exact(x, t_fixed), label=f't = {t_fixed}')

plt.figure(1)
plt.xlabel('Position x')
plt.ylabel('u_exact')
plt.title('Évolution de la chaleur en 1D à t fixé')
plt.legend()


x, t = sp.symbols('x t')
u_ex = (1/np.sqrt(2*np.pi))*sp.exp((-x**2)/2)*(1+t)
du_dx = sp.diff(u_ex, x)
du_dx_2 = sp.diff(du_dx, x)
du_dt = sp.diff(u_ex, t)
f = du_dt - D * du_dx_2 + C * du_dx

def fonction_f(n, L=L, f=f):
    return float(f.subs({x: L, t: n}))




# Boucle temporelle
for n in range(0, nt):
    u_new = u.copy()
    for i in range(1, nx - 1):
        # Calcul avec ajout de F(x) dans l'équation
        u_new[i] = u[i] + D * dt / dx**2 * (u[i + 1] - 2 * u[i] + u[i - 1]) - C * dt / dx * (u[i + 1] - u[i]) + fonction_f(n) * dt# Ajout de la source F(x) ici
    u = u_new.copy()
    u_mat[n, :] = u




# Select a fixed t index
t_index = 50  # Example: middle of the time range
x=np.linspace(0, L, nx)
plt.figure(2)

# Plot u_mat as a function of x for the fixed t
plt.plot(x, u_mat[t_index, :])
plt.plot(x, u_exact(x, t_index), label=f't = {t_index}')

plt.xlabel('Position x')
plt.ylabel('u_mat')
plt.title(f'u_mat as a function of x at t = {t_index}')
plt.show()

