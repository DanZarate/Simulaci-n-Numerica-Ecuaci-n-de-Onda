import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import os

# Crear la carpeta "frames" si no existe
output_folder  = "EcOndaBidimenseional/frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
save = True # Cambiar a True para guardar los frames

# =============================================
# PARÁMETROS DEL PROBLEMA
# =============================================
# PARÁMETROS DEL CANAL
L = 2.0           # Mitad del largo del canal (longitud total = 2L)
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
xi = 0.5#*np.pi        # Parámetro de frecuencia espacial

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
    boundary_grad = 0.0# A * sin_term * np.cos(xi * current_time)  # Condición completa

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
# Condición (ψ = x*z)
psi_n = X * Z.copy()

# Nueva condición inicial (ψ = A sin(ξx) sin(ξz))
#psi_n = A * np.sin(xi * X) * np.sin(xi * Z)

psi_nm1 = np.zeros((Nx, Nz))    # ψ en t-Δt

# Inicialización usando ψ_t(x,z,0) = 0
laplacian_initial = compute_laplacian(psi_n, 0.0)
psi_nm1 = psi_n - 0.5 * dt**2 * gh * laplacian_initial

# =============================================
# SIMULACIÓN
# =============================================
frames = []
times = []  # Lista para almacenar los tiempos de cada frame
# Lista para almacenar todos los índices de frames
all_frame_indices = []
for n in range(Nt):
    current_time = n * dt
    laplacian = compute_laplacian(psi_n, current_time)
    psi_np1 = 2*psi_n - psi_nm1 + dt**2 * gh * laplacian
    
    # Almacenar ambas soluciones
    if n % 10 == 0:
        frames.append((psi_np1.copy(), analytical_solution(current_time)))
        all_frame_indices.append(len(frames)-1)  # Guardar índice del frame
        times.append(current_time)
    # Actualizar iteración
    psi_nm1 = psi_n.copy()
    psi_n = psi_np1.copy()

# =============================================
# SELECCIÓN Y GUARDADO DE FRAMES ALEATORIOS
# =============================================
# Seleccionar 5 frames aleatorios de los disponibles
num_random_frames = 5
frame_indices = sorted(random.sample(all_frame_indices, min(num_random_frames, len(all_frame_indices))))
print("Frames seleccionados:", frame_indices)

# =============================================
# ANIMACIÓN DUAL 2D
# =============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))  # 18" ancho x 4" alto
#fig.suptitle('Comparación de Soluciones 2D', y=1.02)  # Ajustar posición del título

# Configuración común
vmin = np.min(frames)  # Mínimo global en todos los tiempos
vmax = np.max(np.abs(frames))
cmap = 'RdBu'
extent = [-L, L, -d, d]

# Ajustar márgenes y espaciado
#fig.subplots_adjust(left=0.05, right=0.85, wspace=0.25, top=0.83)

# Inicializar plots
im1 = ax1.imshow(frames[0][0].T, origin='lower', extent=extent, 
                cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
im2 = ax2.imshow(frames[0][1].T, origin='lower', extent=extent, 
                cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

# Configurar ejes y colorbars
for ax, title in zip([ax1, ax2], ['Numérica (Nx=201, Nz=101)', 'Analítica (m=20, n=20)']):
    ax.set_xlabel('Coordenada x', fontsize=10)
    ax.set_ylabel('Coordenada z', fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    
    # Añadir colorbar individual con etiqueta
    cbar = fig.colorbar(im1, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Amplitud ψ', rotation=270, fontsize=10, labelpad=15)
    cbar.ax.tick_params(labelsize=8)

# Función de actualización optimizada
def update(frame):
    im1.set_data(frames[frame][0].T)
    im2.set_data(frames[frame][1].T)
    fig.suptitle(f'Evolución temporal: t = {frame*10*dt:.2f} s', fontsize=14)
    return im1, im2

ani = FuncAnimation(fig, update, frames=len(frames), interval=100)

if save:
    # Crear figura para guardado
    for idx in frame_indices:
        update(idx)      # Actualiza la figura al frame idx
        plt.draw()       # Fuerza la actualización de la figura
        # Guarda el frame actual en un archivo PNG
        file_path = os.path.join(output_folder, f"frame_{idx}.png")
        fig.savefig(file_path)
        
    gif_path = os.path.join(output_folder, "animacion.gif")
    ani.save(gif_path, writer='pillow')

#plt.show()

# =============================================
# SIMULACIÓN CON CÁLCULO DE ERROR
# =============================================
error_frames = []
max_errors = []
times = []

for n in range(Nt):
    current_time = n * dt
    # [...] (Código de simulación existente)
    
    if n % 10 == 0:
        # Calcular error absoluto
        error = np.abs(frames[-1][0] - frames[-1][1])  # [numérico - analítico]
        error_frames.append(error)
        max_errors.append(np.max(error))
        times.append(current_time)

# =============================================
# VISUALIZACIÓN DEL ERROR
# =============================================
# Configuración de gráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Análisis de Error')

# 1. Mapa de calor del error en el último paso
im = ax1.imshow(error_frames[-1].T, origin='lower', extent=[-L, L, -d, d],
               cmap='hot', aspect='auto')
fig.colorbar(im, ax=ax1, label='Error absoluto')
ax1.set_title('Distribución espacial del error')
ax1.set_xlabel('x')
ax1.set_ylabel('z')

# 2. Evolución temporal del error máximo
ax2.plot(times, max_errors, 'r-', linewidth=2)
ax2.set_title('Error máximo vs Tiempo')
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Error máximo')
ax2.grid(True)

# 3. Animación de la evolución del error
fig_anim, ax_anim = plt.subplots(figsize=(7,5))
im_anim = ax_anim.imshow(error_frames[0].T, origin='lower', extent=[-L, L, -d, d],
                        cmap='hot', vmin=0, vmax=np.max(max_errors), aspect='auto')
cb = fig_anim.colorbar(im_anim, label='Error absoluto')

def update_anim(frame):
    im_anim.set_data(error_frames[frame].T)
    ax_anim.set_title(f'Error absoluto en t = {times[frame]:.2f} s')
    return im_anim,

ani = FuncAnimation(fig_anim, update_anim, frames=len(error_frames), interval=100)
plt.show()