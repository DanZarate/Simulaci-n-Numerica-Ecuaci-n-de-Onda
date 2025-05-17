import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import os

# Crear la carpeta "frames" si no existe
folder = "EcOndaBidimenseional/Punto1/frames"
if not os.path.exists(folder):
    os.makedirs(folder)

# =============================================
# PARÁMETROS DEL PROBLEMA
# =============================================
# PARÁMETROS DEL CANAL
L = 5.0           # Mitad del largo del canal (longitud total = 2L)
d = 1.0           # Mitad del ancho del canal (ancho total = 2d)
gh = 9.81 * 0.5   # gh = g * profundidad (ej: profundidad = 0.5 m, g=9.81 m/s²)
Nx = 201           # Alta resolución en x (para capturar propagación a lo largo del canal)
Nz = 51            # Resolución moderada en z (paredes laterales)

T_total = 20.0     # Tiempo total de simulación
m_max = 20        # Términos máximos serie analítica
n_max = 20

# =============================================
# DISCRETIZACIÓN ESPACIAL
# =============================================
dx = 2*L / (Nx-1)
dz = 2*d / (Nz-1)
x = np.linspace(-L, L, Nx)
z = np.linspace(-d, d, Nz)
X, Z = np.meshgrid(x, z, indexing='ij')

# =============================================
# PARÁMETROS DEL PROBLEMA
# =============================================
A = 1           # Amplitud de la condición inicial
xi = 0.5*np.pi        # Parámetro de frecuencia espacial

# =============================================
# FUNCIÓN DE FUENTE f(x,t)
# =============================================
def f_source(x, t):
    return 0.0

# =============================================
# SOLUCIÓN ANALÍTICA (VECTORIZADA)
# =============================================
def analytical_solution(t):
    psi = np.zeros_like(X)
    for m in range(1, m_max+1):
        for n in range(1, n_max+1):
            coeff = (16 * d * L * (np.cos(np.pi * n) - 1) * (np.cos(np.pi * m) - 1)) 
            coeff /= (n**2 * m**2 * np.pi**4)
            
            spatial = np.cos((m*np.pi)/(2*L)*(X + L)) * np.cos((n*np.pi)/(2*d)*(Z + d))
            lambda_mn = (np.pi/2) * np.sqrt((m/L)**2 + (n/d)**2) * np.sqrt(gh)
            
            psi += coeff * spatial * np.cos(lambda_mn * t)
    return psi

# =============================================
# FUNCIÓN LAPLACIANO (CONDICIONES NEUMANN)
# =============================================
def compute_laplacian(psi, current_time):
    laplacian = np.zeros_like(psi)
    
    # Condición izquierda (ψ_x(-L) = 0)
    #laplacian[0, :] = (2*psi[1, :] - 2*psi[0, :])/dx**2

    # Condición izquierda (x=-L): ∂ψ/∂x = sin(πz/(2d))cos(ξt)
    sin_term = np.sin(np.pi * Z[0, :]/(2*d))  # Término espacial (z)
    boundary_grad = 0.0#A * sin_term * np.cos(xi * current_time)  # Condición completa

    # Aproximación de segunda derivada usando condición de Neumann no homogénea
    laplacian[0, :] = (2*psi[1, :] - 2*psi[0, :] - 2*dx*boundary_grad)/dx**2

    # Condición derecha (x=L): ∂ψ/∂x = 0 (Neumann homogénea)
    laplacian[-1, :] = (2*psi[-2, :] - 2*psi[-1, :])/dx**2
    # Condición derecha (ψ_x(L) = A sin(ξt))
    #laplacian[-1, :] = (2*psi[-2, :] - 2*psi[-1, :] + 2*dx*A*np.cos(xi*current_time)*np.sin((np.pi*Z)/(2*d)))/dx**2

    # Derivadas en x
    laplacian[1:-1, :] = (psi[2:, :] - 2*psi[1:-1, :] + psi[:-2, :])/dx**2    
    
    # Derivadas en z (ψ_z(±d) = 0)
    laplacian[:, 1:-1] += (psi[:, 2:] - 2*psi[:, 1:-1] + psi[:, :-2])/dz**2
    laplacian[:, 0] += (2*psi[:, 1] - 2*psi[:, 0])/dz**2    # z = -d
    laplacian[:, -1] += (2*psi[:, -2] - 2*psi[:, -1])/dz**2 # z = d
    
    return laplacian


# =============================================
# INICIALIZACIÓN NUMÉRICA
# =============================================
dt = 0.9 * min(dx, dz)/np.sqrt(2*gh)
Nt = int(T_total/dt) + 1

#psi_n = np.zeros((Nx, Nz))    # ψ(x,z,0) = 0
# Condición original (ψ = x*z)
psi_n = X * Z.copy()

# Nueva condición inicial (ψ = A sin(ξx) sin(ξz))
# psi_n = A * np.sin(xi * X) * np.sin(xi * Z)

psi_nm1 = np.zeros((Nx, Nz))    # ψ en t-Δt

#f0 = f_source(x, 0)[:, np.newaxis] * np.ones(Nz)  # Fuente en t=0

# Inicialización usando ψ_t(x,z,0) = 0
laplacian_initial = compute_laplacian(psi_n, 0.0)
psi_nm1 = psi_n - 0.5 * dt**2 * (gh * laplacian_initial)

energy_history = []
control_history = []

# =============================================
# SIMULACIÓN
# =============================================
frames = []
for n in range(Nt):
    current_time = n * dt

    #f = f_source(x, current_time)[:, np.newaxis] * np.ones(Nz)
    
    laplacian = compute_laplacian(psi_n, current_time)
    # Paso de integración temporal
    #psi_np1 = 2*psi_n - psi_nm1 + dt**2 * gh * laplacian
    # Paso temporal con término de forzamiento
    psi_np1 = 2*psi_n - psi_nm1 + dt**2 * (gh * laplacian)
    
    # Almacenar ambas soluciones
    if n % 10 == 0:
        frames.append((psi_np1.copy(), analytical_solution(current_time)))#psi_np2.copy()))#
    
    # Actualizar iteración
    psi_nm1 = psi_n.copy()
    psi_n = psi_np1.copy()


# =============================================
# ANIMACIÓN DUAL
# =============================================
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Configuración común
vmin = int(np.min(frames))  # Mínimo global en todos los tiempos
vmax = int(np.max(np.abs(frames)))

def update(frame):
    ax1.clear()
    ax2.clear()
    
    # Solución numérica
    ax1.plot_surface(X, Z, frames[frame][0], cmap='viridis')
    ax1.set_title(f"Solución Numérica\n t = {frame*10*dt:.2f} s")
    ax1.set_zlim(-vmax, vmax)
    
    # Solución analítica
    ax2.plot_surface(X, Z, frames[frame][1], cmap='plasma')
    ax2.set_title(f"Solución Analítica\n t = {frame*10*dt:.2f} s")
    ax2.set_zlim(-vmax, vmax)
    
    return ax1, ax2

ani = FuncAnimation(fig, update, frames=len(frames), interval=100)
plt.show()
