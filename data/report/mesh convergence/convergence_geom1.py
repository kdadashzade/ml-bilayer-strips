import matplotlib.pyplot as plt
import numpy as np

# Add mesh convergence data here
data = [
    (2.0, 50, 35.4286),
    (1.5, 66, 35.4218), 
    (1.0, 100,  35.4435),
    (0.8, 126,  35.4552),
    (0.6, 166,  35.4640),
    (0.4, 250, 35.4912),
    (0.3, 334, 35.5091),
    (0.275, 364, 35.5139),
    (0.25, 800, 35.4790),
    (0.20, 1000, 35.4869),
    (0.15, 1998, 35.4874),
    (0.1, 4000, 35.4882)
]

title       = "Mesh Convergence - Geometry 1"
output_file = "mesh_convergence1.png"
use_abs     = True
# ============================================================

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'axes.labelcolor': '#222222',
    'xtick.color': '#444444',
    'ytick.color': '#444444',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

mesh_size = np.array([d[0] for d in data])
result    = np.array([d[2] for d in data])
y         = np.abs(result) if use_abs else result

sort_idx  = np.argsort(mesh_size)
converged = np.mean(y[sort_idx][:4])

C_LINE   = '#3A8A5C'
C_ACCENT = '#C8553D'

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FFFFFF')
ax.grid(True, which='major', linestyle='-',  linewidth=0.6, color='#DDDDDD', zorder=0)
ax.tick_params(which='both', length=4)
ax.invert_xaxis()

ax.axhline(converged, color=C_ACCENT, linestyle='--', linewidth=1.5,
           zorder=2, label=f'Converged ≈ {converged:.3f} mm')

ax.plot(mesh_size, y, '-', color=C_LINE, linewidth=2, alpha=0.85, zorder=3)
ax.scatter(mesh_size, y, s=60, color=C_LINE, edgecolor='white',
           linewidth=1.5, zorder=4, label='Recorded values')

ylabel = ('Result' if use_abs else 'Result') + ' (mm)'
ax.set_xlabel('Element size (mm)', fontweight='medium')
ax.set_ylabel(ylabel, fontweight='medium')

ax.set_title(title, fontsize=15, fontweight='bold', pad=12, loc='left')

ax.legend(loc='best', frameon=True, framealpha=0.95,
          edgecolor='#CCCCCC', fontsize=9.5)

fig.tight_layout()
fig.savefig(output_file, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()