# EDP-Project


# Résolution d'Équations aux Dérivées Partielles (EDP) en Python

## Introduction
Ce projet présente des solutions pour les équations aux dérivées partielles (EDP) en dimensions 1 et 2 en utilisant la méthode des différences finies en Python. Nous abordons l'équation de la chaleur en 1D et l'équation de Laplace en 2D.

## Pré-requis
- Python 3.10
- Bibliothèques Python : numpy, matplotlib

Vous pouvez installer les bibliothèques nécessaires en utilisant pip :
```bash
pip install numpy matplotlib
```

## Équations aux Dérivées Partielles (EDP)

### Dimension 1 : Équation de la chaleur
L'équation de la chaleur en une dimension décrit la distribution de la température dans une barre au fil du temps :
$$ \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} $$

### Dimension 2 : Équation de Laplace
L'équation de Laplace en deux dimensions est utilisée pour modéliser des phénomènes tels que le potentiel électrique et les champs de température stationnaires :
$$ \nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0 $$

## Implémentation en Python

### Équation de la chaleur en 1D
Voici un exemple de code pour résoudre l'équation de la chaleur en 1D :

```python
import numpy as np
import matplotlib.pyplot as plt

# Paramètres
L = 10.0     # Longueur de la barre
T = 2.0      # Temps total
nx = 100     # Nombre de points en espace
nt = 1000    # Nombre de points en temps
alpha = 0.01 # Diffusivité thermique

dx = L / (nx - 1)   # Discrétisation en espace
dt = T / nt         # Discrétisation en temps
u = np.zeros(nx)    # Initialisation de la température
u[int(nx/2)] = 1    # Condition initiale: un pic de chaleur au centre

# Matrice de solution
u_mat = np.zeros((nt, nx))
u_mat[0, :] = u

# Boucle temporelle
for n in range(1, nt):
    u_new = u.copy()
    for i in range(1, nx-1):
        u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] - u[i-1])
    u = u_new.copy()
    u_mat[n, :] = u

# Affichage
plt.imshow(u_mat, extent=[0, L, T, 0], aspect='auto', cmap='hot')
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Temps t')
plt.title('Évolution de la chaleur en 1D')
plt.show()
```

### Équation de Laplace en 2D
Voici un exemple de code pour résoudre l'équation de Laplace en 2D :

```python
import numpy as np
import matplotlib.pyplot as plt

# Paramètres
Lx = 10.0     # Longueur en x
Ly = 10.0     # Longueur en y
nx = 50       # Nombre de points en x
ny = 50       # Nombre de points en y

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
u = np.zeros((ny, nx))

# Conditions aux bords
u[0, :] = 100  # Bord supérieur
u[:, 0] = 0    # Bord gauche
u[:, -1] = 0   # Bord droit
u[-1, :] = 0   # Bord inférieur

# Boucle de relaxation de Gauss-Seidel
tolerance = 1e-6
error = 1.0

while error > tolerance:
    u_new = u.copy()
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
    error = np.linalg.norm(u_new - u)
    u = u_new.copy()

# Affichage
plt.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='hot')
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Position y')
plt.title('Solution de l\'équation de Laplace en 2D')
plt.show()
```

## Conclusion
Ce projet présente des exemples de résolution d'EDP en utilisant la méthode des différences finies en Python. Vous pouvez utiliser ces exemples comme point de départ pour explorer et résoudre d'autres EDP dans vos propres projets.

## Licence
Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus d'informations.

