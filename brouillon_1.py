import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Paramètres
L = 1.0     # Longueur de la barre
T = 2.0      # Temps total
nx = 100
D = 0.01 
dx = L / (nx - 1)     # Nombre de points en espace
dt = dx**2/(2*D)   # Pas de temps
    
C =0.03        # Coefficient pour le terme advection
# Discrétisation en espace
nt = int(T / dt)    # Discrétisation en temps
x=np.zeros(nx)
# Initialisation des matrices de température
u = np.zeros((nt, nx))  # Initialisation de la température pour tous les temps
for i in range(nx):
    x[i]=i*dx

#Permet de calculer f(x,t) pour u quelconque par le calcul formel, il suffit de modifier u_ex dans la fonction
def determinate_f():
    y, t = sp.symbols('y t') #les variables sont y et t pour la résolution formelle 
    u_ex=sp.sin(np.pi*y)*(1+t)
    #u_ex = (1/np.sqrt(2*np.pi))*sp.exp(-(y**2)/2)*(1+t) #gaussienne avec mu=0 et sigma=1
    du_dx = sp.diff(u_ex, y)
    du_dx_2 = sp.diff(du_dx, y)
    du_dt = sp.diff(u_ex, t)
    f = du_dt - D * du_dx_2 + C * du_dx
    print(f)
    return f

y, t = sp.symbols('y t')
f=determinate_f()

def function_f(x, tps, know_f=True, f=f):
    if know_f:
        """Terme source."""
        return np.sin(np.pi * x) + D * (np.pi**2) * np.sin(np.pi * x) * (1 + tps) + C * np.pi * np.cos(np.pi * x) * (1 + tps)
    else:
        return float(f.subs({y: x, t: tps}))

# Condition initiale: distribution initiale constante
u[0, :] = np.sin(np.pi*x)  # constante sur la longueur de la barre

# Conditions aux limites
def apply_boundary_conditions(u_n):
    """Applique les conditions aux limites Dirichlet."""
    u_n[0] = 0  # Dirichlet à gauche
    u_n[-1] = 0 # Dirichlet à droite
    return u_n

# Boucle temporelle
for n in range(0, nt - 1):
    for i in range(1, nx - 1):
        print(n,i)
        convection = -C * (u[n, i + 1] - u[n, i - 1]) / (2 * dx)  # Formule centrée pour le terme de convection
        diffusion = D * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / dx**2
        source = function_f(i*dx, n * dt)
        u[n + 1, i] = u[n, i] + dt * (convection + diffusion + source)
    u[n + 1, :] = apply_boundary_conditions(u[n + 1, :])


def u_exact(x, t):
    #return (1/np.sqrt(2*np.pi))*np.exp(-(x**2)/2)*(1+t)
    return np.sin(np.pi*x)*(1+t)
# Calcul de l'erreur en norme L2
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
u_exact_mat = np.zeros((nt, nx))
for n in range(nt):
    for i in range(nx):
        u_exact_mat[n, i] = u_exact(x[i], t[n])


# Affichage des résultats
x = np.linspace(0, L, nx)  # Ajout du maillage spatial
for n in range(0, nt, nt // 5):
    plt.plot(x, u[n, :], label=f"t={n * dt:.2f}")

plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.title("Équation de la chaleur avec convection et diffusion")
plt.show()
L2_error = np.linalg.norm(u-u_exact_mat)
print(f"Erreur en norme L2: {L2_error}")
L2_relative=np.linalg.norm(u-u_exact_mat)/np.linalg.norm(u_exact_mat)
print(f"Erreur relative L2: {(L2_relative)*100:.4f}%")

# Affichage
plt.imshow(u, extent=[0, L, T, 0], aspect='auto', cmap='hot')
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Temps t')
plt.title('Évolution de la chaleur en 1D avec source F(x)')
plt.show()
L2_error = np.sqrt(np.sum((u - u_exact_mat)**2, axis=1) * dx)
time = np.arange(0, nt) * dt

# Tracer l'erreur L2 en fonction du temps
plt.plot(time, L2_error, label="Erreur L2")
plt.xlabel("Temps")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Erreur L2")
plt.legend()
plt.grid()
plt.show()










