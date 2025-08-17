import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N = 15                          # number of nodes
np.random.seed(0)
positions = np.random.rand(N, 2) * 10  # random positions

# Overlap (semantic similarity proxy)
O = np.random.rand(N, N)
O = (O + O.T) / 2  # symmetric
np.fill_diagonal(O, 0)

# Coupling modulation parameters
alpha, beta = 2.0, 1.5

def coupling_matrix(O):
    return (alpha * O) / (1 + beta * O**2)

# Initial phases
phases = np.random.rand(N) * 2 * np.pi

# Base frequency and adaptation
omega0 = 0.05
gamma = 0.5

def global_coherence(phases):
    return np.abs(np.mean(np.exp(1j * phases)))

# Animation update function
def update(num, phases, scat, lines):
    K = coupling_matrix(O)

    # compute global coherence
    C = global_coherence(phases)
    omega = omega0 * (1 + gamma * C)

    # Kuramoto-like update
    for i in range(N):
        interaction = np.sum(K[i, j] * np.sin(phases[j] - phases[i]) for j in range(N))
        phases[i] += omega + interaction * 0.01

    # update node visuals
    x = np.cos(phases) + positions[:, 0]
    y = np.sin(phases) + positions[:, 1]
    scat.set_offsets(np.c_[x, y])

    # update connections with sigmoid thickness
    for l, (i, j) in zip(lines, np.argwhere(O > 0.6)):
        thickness = 1 / (1 + np.exp(-5 * (K[i, j] - 0.5)))  # sigmoid map
        l.set_data([x[i], x[j]], [y[i], y[j]])
        l.set_linewidth(thickness * 3)
        l.set_alpha(0.6)
    return scat, lines

# Plot setup
fig, ax = plt.subplots(figsize=(6, 6))
scat = ax.scatter(positions[:, 0], positions[:, 1], s=80, c='royalblue')

lines = [ax.plot([], [], 'k-', alpha=0.5)[0] for _ in np.argwhere(O > 0.6)]

ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.axis('off')

ani = animation.FuncAnimation(fig, update, frames=200, fargs=(phases, scat, lines), interval=50, blit=False)
plt.show()
