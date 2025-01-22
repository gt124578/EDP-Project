import numpy as np
import matplotlib.pyplot as plt

# PHYSICAL PARAMETERS
K = 0.5     #Diffusion coefficient
L = 1.0     #Domain size
Time = 0.1  #Integration time

# NUMERICAL PARAMETERS
NX = 100    #Number of grid points
NT = 1000   #Number of time steps

dx = L/(NX-1)  #Grid step (space)
dt = Time/NT   #Grid step (time)


### MAIN PROGRAM ###

# Initialisation
x = np.linspace(0.0,1.0,NX)
T = np.sin(2*np.pi*x)
RHS = np.zeros((NX))

plt.figure()

# Main loop
for n in range(0,NT):

   for j in range (1, NX-1):
      RHS[j] = dt*K*(T[j-1]-2*T[j]+T[j+1])/(dx**2)

   for j in range (1, NX-1):
      T[j] += RHS[j]

#Plot every 100 time steps
   if (n%100 == 0):
      plotlabel = "t = %1.2f" %(n * dt)
      plt.plot(x,T, label=plotlabel,color = plt.get_cmap('copper')(float(n)/NT))
      

plt.xlabel(u'$x$', fontsize=26)
plt.ylabel(u'$T$', fontsize=26, rotation=0)
plt.title(u'Equation de la chaleur 1D')




# # Définition de l'exacte solution u_exact
# def u_exact(x, t, D=K):
#     return np.sin(np.pi * x)*(1 + t) 

# for t_fixed in range(0,NT):
#     if t_fixed%100==0:
#         # Affichage
#         plt.plot(x, u_exact(x, t_fixed), label=f't = {float(t_fixed)/NT}')

plt.legend()
plt.show()



# # Initialisation
# x = np.linspace(0.0,1.0,nx)
# T = np.sin(2*np.pi*x)
# # Main loop
# for n in range(0,nt):

#    for j in range (1, nx-1):
#       T[j] += dt*D*(T[j-1]-2*T[j]+T[j+1])/(dx**2)+ fonction_f(n) * dt

# # Select a fixed t index
# t_index = 50  # Example: middle of the time range
# x=np.linspace(0, L, nx)
# plt.figure(2)

# # Plot u_mat as a function of x for the fixed t
# plt.plot(x, T[x])
# plt.plot(x, u_exact(x, t_index), label=f't = {t_index}')

# plt.xlabel('Position x')
# plt.ylabel('u_mat')
# plt.title(f'u_mat as a function of x at t = {t_index}')
# plt.show()
