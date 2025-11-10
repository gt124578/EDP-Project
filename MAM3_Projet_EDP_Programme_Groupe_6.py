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
    #print(f)
    return f

y, t = sp.symbols('y t')
f=determinate_f()

#si know_f=False cela calcul f de façon formelle en partant de u exact à définir (ce qui rallonge énormément le temps de calcul)
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
        #print(n,i)
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
plt.title("Evolution de la température en fonction de x avec l'équation de la chaleur en dimension 1 pour différents t")
plt.show()
L2_error = np.linalg.norm(u-u_exact_mat)
print(f"Erreur en norme L2 en 1D: {L2_error}")
L2_relative=np.linalg.norm(u-u_exact_mat)/np.linalg.norm(u_exact_mat)
print(f"Erreur relative L2 en 1D: {(L2_relative)*100:.4f}%")

# Affichage
plt.imshow(u, extent=[0, L, T, 0], aspect='auto', cmap='hot')
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Temps')
plt.title('Évolution de la chaleur en 1D avec source f(x,t)')
plt.show()
L2_error = np.sqrt(np.sum((u - u_exact_mat)**2, axis=1) * dx)
time = np.arange(0, nt) * dt

#Etude de l'évolution de L2 pour différents dt

dt_factors = [0.1, 0.5, 1.0]  # Facteurs de variation pour dt

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

plt.xlabel("Temps")
plt.ylabel("Erreur L2")
plt.title("Évolution de l'erreur L2 pour différents dt")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()







#Etude de l'évolution de l'erreur L2 pour différents dx


# Différentes valeurs de nx
nx_values = [50, 100, 200, 400]  # Résolutions spatiales


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
            source = function_f(i * dx, n * dt)
            u[n + 1, i] = u[n, i] + dt * (convection + diffusion + source)
        u[n + 1, :] = apply_boundary_conditions(u[n + 1, :])

    t = np.linspace(0, T, nt)
    u_exact_mat = np.array([[u_exact(x[i], t[n]) for i in range(nx)] for n in range(nt)])
    L2_error = np.sqrt(np.sum((u - u_exact_mat)**2, axis=1) * dx)  # Erreur en norme L2 pour chaque instant
    return x, t, L2_error

# Résolution pour chaque nx et tracé des erreurs
plt.figure()
for nx in nx_values:
    x, t, L2_error = solve_heat_eq(nx, L, T, D, C)
    plt.plot(t, L2_error, label=f"dx = {L / (nx - 1):.4f}")

plt.xlabel("Temps")
plt.ylabel("Erreur L2")
plt.title("Évolution de l'erreur L2 pour différents dx")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()






#Etude de l'équation de la chaleur en dimension 2




# Paramètres
Lx, Ly = 1.0, 1.0  # Dimensions du domaine
T = 2.0  # Temps total
nx, ny = 50, 50  # Nombre de points en espace
D = 0.01
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = min(dx, dy)**2 / (4 * D)  # Pas de temps
nt = int(T / dt)  # Discrétisation en temps
Cx = 0.03  # Coefficient de convection selon x
Cy = 0.02  # Coefficient de convection selon y
# Initialisation des matrices de température
u = np.zeros((nt, nx, ny))
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)
X, Y = np.meshgrid(x, y)


#Permet de calculer f(x,y,t) pour u quelconque par le calcul formel, il suffit de modifier u_ex dans la fonction
def determinate_f_2D():
    a, b, u = sp.symbols('a b u') #les variables sont a, b et u pour la résolution formelle 
    u_ex=sp.sin(sp.pi * a) * sp.sin(sp.pi * b) * (1 + u)
    du_dx = sp.diff(u_ex, a)
    du_dy = sp.diff(u_ex, b)
    du_dx_2 = sp.diff(du_dx, a)
    du_dy_2 = sp.diff(du_dx, b)
    du_dt = sp.diff(u_ex, u)
    f = du_dt - D * (du_dx_2+du_dy_2) + Cx * du_dx + Cy * du_dy
    #print(f)
    return f

# a, b, u = sp.symbols('a b u')
# f=determinate_f_2D()

def function_f(x, y, t):
    """Terme source."""
    return D*np.pi**2*(t + 1)*np.sin(np.pi*x)*np.sin(np.pi*y) + Cy*np.pi*(t + 1)*np.sin(np.pi*x)*np.cos(np.pi*y) + Cx*np.pi*(t + 1)*np.sin(np.pi*y)*np.cos(np.pi*x) - D*np.pi**2*(t + 1)*np.cos(np.pi*x)*np.cos(np.pi*y) + np.sin(np.pi*x)*np.sin(np.pi*y)



# Condition initiale
u[0, :, :] = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Conditions aux limites
def apply_boundary_conditions_2D(u_n):
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
            convection_x = -Cx * (u[n, i + 1, j] - u[n, i - 1, j]) / (2 * dx)  # Convection selon x
            convection_y = -Cy * (u[n, i, j + 1] - u[n, i, j - 1]) / (2 * dy)  # Convection selon y
            source = function_f(x[i], y[j], t[n])
            u[n + 1, i, j] = u[n, i, j] + dt * (diffusion + convection_x + convection_y + source)
    u[n + 1, :, :] = apply_boundary_conditions_2D(u[n + 1, :, :])

#U(x,y,t) exact en 2D 
def u_exact(x, y, t, D=D):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * (1 + t)

u_exact_mat = np.zeros((nt, nx, ny))
for n in range(nt):
    for i in range(nx):
        for j in range(ny):
            u_exact_mat[n, i, j] = u_exact(x[i], y[j], t[n])

L2_error = np.sqrt(np.sum((u - u_exact_mat)**2) * dx * dy * dt)
print(f"Erreur relative L2 en 2D: {(L2_error)*100:.4f}%")

#Affichage des résultats pour t=1
plt.contourf(X, Y, u[1, :, :], cmap='hot')
plt.colorbar()
plt.title(f"Équation de la chaleur en 2D à t={1:.2f}")
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Variation de température en fonction de x pour t donnés
for n in range(0, nt, nt // 5):
    plt.plot(x, u[n, :, ny//2], label=f"t={n * dt:.2f}")

plt.xlabel("Position x")
plt.ylabel("Température u(x, y=L/2, t)")
plt.legend()
plt.title("Evolution de la température en fonction de x avec l'équation de la chaleur en dimension 2 (y=L/2) pour différents t")
plt.show()

# Tracer l'erreur L2 en fonction du temps
L2_error_time=np.sqrt(np.sum((u - u_exact_mat)**2, axis=(1, 2)) * dx * dy)

plt.figure()
plt.plot(t, L2_error_time, label="Erreur L2")
plt.xlabel("Temps")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Erreur L2")
plt.legend()
plt.title('Erreur L2 de u(x,y,t) par rapport à la solution exacte en fonction du temps')
plt.grid()
plt.show()
