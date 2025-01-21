from sympy.solvers.pde import pdsolve
from sympy import Function, Eq
from sympy.abc import x, t

u = Function('u')(x,t)

# Paramètres
D = 0.01  # Diffusivité thermique
C = 3     # Coefficient pour le terme advection

ux = u.diff(x)
uxx=ux.diff(x)
ut = u.diff(t)

eq = Eq((2*(ux)) + (3*(ut)+u), 0)

print(pdsolve(eq))

#calcul edp_exacte problème avec uxx
