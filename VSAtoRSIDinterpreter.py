import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from scipy.fft import fft, fftfreq

# -- Step 0: Setup dummy RSID simulation (minimal) --
# Simulate 3 attractors in RSID for 3 simple VSA seeds

def simulate_rsid_attractor(seed_vector, num_osc=5, timesteps=200):
    np.random.seed(int(np.sum(seed_vector)*1000)%10000)
    phases = np.random.rand(num_osc, timesteps) * 2*np.pi
    # Introduce coherence if seed > threshold
    if seed_vector.sum() > 0.5:
        # Make oscillators align in second half
        phases[:, timesteps//2:] = phases[0, timesteps//2:] + 0.1 * np.random.randn(num_osc, timesteps//2)
    return phases

# -- Step 1: Generate VSA seeds (3 concepts) --
# Simple 4-dim VSA examples
vsa_seeds = {
    'A': np.array([1, 0, 1, 0]),
    'B': np.array([0, 1, 1, 0]),
    'C': np.array([0, 0, 1, 1]),
}

# -- Step 2: Collect fingerprints --
# Fingerprint = [mean_phase_dispersion, coherence_level, dominant_freq]
def extract_fingerprint(phases, dt=0.01):
    num_osc, T = phases.shape
    sin_ph = np.sin(phases)
    coherence = np.abs(np.mean(np.exp(1j*phases), axis=0)).mean()
    # Phase dispersion: average std dev across oscillators
    phase_disp = np.mean(np.std(phases, axis=0))
    # FFT of coherence over time
    coh_ts = np.abs(np.mean(np.exp(1j*phases), axis=0))
    yf = fft(coh_ts)
    xf = fftfreq(T, dt)[:T//2]
    dominant_freq = xf[np.argmax(np.abs(yf[:T//2]))]
    return np.array([phase_disp, coherence, dominant_freq])

X = []
y_vsa = []
labels = []
for label, v in vsa_seeds.items():
    phases = simulate_rsid_attractor(v)
    fp = extract_fingerprint(phases)
    X.append(fp)
    y_vsa.append(v)
    labels.append(label)
X = np.vstack(X)
y_vsa = np.vstack(y_vsa)

# -- Step 3: Train Ridge decoder (fingerprint -> VSA) --
decoder = Ridge(alpha=1.0)
decoder.fit(X, y_vsa)

# -- Step 4: Train classifier from fingerprint to labels (symbolic) --
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)
clf.fit(X, labels)

# -- Step 5: Demo on new noisy seeds --
print("=== Demo on new noisy seeds ===")
test_seeds = {
    'A_noisy': vsa_seeds['A'] + np.random.randn(4)*0.1,
    'B_noisy': vsa_seeds['B'] + np.random.randn(4)*0.1,
    'C_noisy': vsa_seeds['C'] + np.random.randn(4)*0.1,
}
for tlabel, tv in test_seeds.items():
    phases = simulate_rsid_attractor(tv)
    fp = extract_fingerprint(phases)
    v_pred = decoder.predict(fp.reshape(1,-1))[0]
    label_pred = clf.predict(fp.reshape(1,-1))[0]
    print(f"{tlabel} → fingerprint={fp.round(3)}")
    print(f"  Decoded VSA ≈ {v_pred.round(2)}, Predicted label: {label_pred}")
