import matplotlib.pyplot as plt
import numpy as np

# Depth values
depth = np.arange(0, 2892, 1)

# Q_mu values
Q_mu = np.piecewise(
    depth, [depth <= 80, (depth > 80) & (depth <= 220), (depth > 220) & (depth <= 670), depth > 670], [600, 80, 143, 312]
)


# Plot Q_mu

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(Q_mu, depth, color=(66.0 / 255.0, 142.0 / 255.0, 14.0 / 255.0), linewidth=2)
plt.title("$Q_\\mu$")
plt.ylabel("Depth (km)")
plt.gca().invert_yaxis()
plt.grid()

# Plot Eta
plt.subplot(1, 2, 2)
plt.semilogx(
    [3 * 10**21, 3 * 10**21],
    [100, 670],
    color=(236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0),
    linestyle=":",
    linewidth=2,
    label="Model 0: uniform anelastic properties",
)
plt.semilogx([3 * 10**20, 3 * 10**20], [100, 670], color=(236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0), linewidth=2)
plt.semilogx([3 * 10**20, 3 * 10**21], [670, 670], color=(236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0), linewidth=2)
plt.semilogx(
    [3 * 10**21, 3 * 10**21], [670, 2891], color=(236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0), linewidth=2, label="Model 1"
)

plt.semilogx(
    [3 * 10**18, 3 * 10**18], [100, 250], color=(236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0), linestyle="--", linewidth=2
)
plt.semilogx(
    [3 * 10**19, 3 * 10**19], [100, 250], color=(236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0), linestyle="--", linewidth=2
)
plt.semilogx(
    [3 * 10**18, 3 * 10**20], [250, 250], color=(236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0), linestyle="--", linewidth=2
)
plt.semilogx(
    [3 * 10**20, 3 * 10**21], [670, 670], color=(236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0), linestyle="--", linewidth=2
)
plt.semilogx(
    [3 * 10**21, 3 * 10**21],
    [670, 2891],
    color=(236.0 / 255.0, 174.0 / 255.0, 24.0 / 255.0),
    linestyle="--",
    linewidth=2,
    label="Model 2: with Burgers in the asthen.",
)
plt.title("$\\eta$ (Pa.s)")
plt.grid()
plt.legend()

# Set log scale for Eta plot
plt.gca().invert_yaxis()

plt.show()
