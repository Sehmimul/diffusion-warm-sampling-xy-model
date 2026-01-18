
import matplotlib.pyplot as plt 
import numpy as np

beta = 1
L = 8

samples = np.load('sandbox/assets/samples_beta_{}_L_{}.npy'.format(beta,L))
samples = samples.reshape((samples.shape[0], L,L))

print(samples.shape)

cmap_choice = 'twilight'

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

for i, ax in enumerate(axes.flat[:10]):
    # Plot heatmap
    i = i + 3000
    im = ax.imshow(samples[i], cmap=cmap_choice, vmin=-np.pi, vmax=np.pi)
    ax.set_title(f'Set {i}', fontsize=12)
    ax.axis('off')
    
    # Overlay arrows for XY spins
    Lx, Ly = samples[i].shape
    X, Y = np.meshgrid(np.arange(Ly), np.arange(Lx))
    U = np.cos(samples[i])
    V = np.sin(samples[i])
    
    ax.quiver(X, Y, U, V, scale=12, color='black', width=0.008)

# Adjust layout for colorbar
fig.subplots_adjust(right=0.85, wspace=0.2, hspace=0.3)

# Add a colorbar
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Angle (radians)', fontsize=14)
cbar.ax.tick_params(labelsize=12)  # change font size for colorbar ticks
plt.savefig("assets/mala_visualiza_beta_{}_L_{}.png".format(beta,L))
