import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, ConnectionPatch
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq

# --- Simulation Parameters ---
total_time = 35.0 # Increased total time for a cleaner signal
time_steps = 2000 # Increased time steps for better resolution
dt = total_time / time_steps
t = np.linspace(0, total_time, time_steps)
training_duration_river = 10.0
training_duration_financial = 10.0
pause_duration = 5.0
probe_duration = 10.0
context_switch_time_1 = training_duration_river
context_switch_time_2 = training_duration_river + training_duration_financial + pause_duration

# --- Hebbian Rule and Forgetting ---
eta = 0.20
alpha_forgetting = 0.8

# --- Oscillator Definitions with Corrected Balanced Omegas ---
# Frequencies for 'river bank' cluster are tightly clustered around 7Hz
# Frequencies for 'financial bank' cluster are tightly clustered around 3Hz
# The shared 'bank' node now has a frequency aligned with the financial cluster.
balanced_omega = {
    'bank': 3.1 * 2 * np.pi, 'loan': 3.0 * 2 * np.pi, 'water': 7.1 * 2 * np.pi,
    'swim': 6.9 * 2 * np.pi, 'money': 2.9 * 2 * np.pi,
}

omegas_to_use = balanced_omega
phases = {
    'bank': 0.0, 'loan': np.pi / 4, 'water': np.pi / 2, 'swim': np.pi,
    'money': -np.pi / 4,
}

initial_coupling_strength = 0.001
coupling_strengths = {
    (n1, n2): initial_coupling_strength for n1 in omegas_to_use for n2 in omegas_to_use if n1 != n2
}

# --- Context-based Attention Mechanism for TRAINING ONLY ---
def get_attention_for_training(t_current):
    if t_current < training_duration_river:
        # Context 1: "River Bank"
        return {
            'bank': {'water': 1.0, 'swim': 1.0},
            'loan': {}, 'water': {'bank': 1.0},
            'swim': {'bank': 1.0}, 'money': {},
        }
    elif t_current < training_duration_river + training_duration_financial:
        # Context 2: "Financial Bank"
        return {
            'bank': {'loan': 1.0, 'money': 1.0},
            'loan': {'bank': 1.0, 'money': 1.0},
            'water': {}, 'swim': {},
            'money': {'bank': 1.0, 'loan': 1.0},
        }
    else:
        return {}

# --- Simulation Runner Function ---
def run_simulation(initial_phases, initial_couplings, t_start, t_end, use_attention=False, omega_set=balanced_omega):
    local_phases = initial_phases.copy()
    local_couplings = initial_couplings.copy()
    
    local_all_phases = {key: [] for key in omega_set.keys()}
    local_all_couplings = {key: [] for key in initial_couplings.keys()}
    
    local_all_coherence_river = []
    local_all_coherence_financial = []

    # Store phase differences to be used for the new FFT analysis
    local_all_phase_diff_river = []
    local_all_phase_diff_financial = []
    
    start_index = int(t_start / dt)
    end_index = int(t_end / dt)
    
    for i in range(start_index, end_index):
        t_current = t[i]
        new_phases = local_phases.copy()
        
        if use_attention:
            attention_matrix = get_attention_for_training(t_current)
            for name_i in omega_set.keys():
                for name_j in omega_set.keys():
                    if name_i != name_j:
                        coupling_key = (name_i, name_j)
                        coupling_val = local_couplings.get(coupling_key, 0)
                        
                        is_in_context = attention_matrix.get(name_i, {}).get(name_j, 0) == 1 or \
                                        attention_matrix.get(name_j, {}).get(name_i, 0) == 1
                        
                        phase_diff = local_phases[name_j] - local_phases[name_i]
                        
                        if is_in_context:
                            dJ_dt = eta * (np.sin(phase_diff))**2 - (alpha_forgetting / 10.0) * coupling_val
                        else:
                            dJ_dt = -alpha_forgetting * coupling_val
                        
                        local_couplings[coupling_key] = max(0, coupling_val + dJ_dt * dt)
        
        for name_i in omega_set.keys():
            total_pull = 0
            for name_j in omega_set.keys():
                if name_i != name_j:
                    coupling_key = (name_i, name_j)
                    coupling_val = local_couplings.get(coupling_key, 0)
                    phase_diff = local_phases[name_j] - local_phases[name_i]
                    total_pull += coupling_val * np.sin(phase_diff)
                    
            d_theta_i_dt = omega_set[name_i] + total_pull
            new_phases[name_i] += d_theta_i_dt * dt
        
        local_phases = new_phases

        for key in local_phases.keys():
            local_all_phases[key].append(local_phases[key])
        
        for coupling_key, coupling_val in local_couplings.items():
            local_all_couplings[coupling_key].append(coupling_val)
        
        cluster_river = [local_phases['bank'], local_phases['water'], local_phases['swim']]
        r_river = np.abs(np.sum(np.exp(1j * np.array(cluster_river)))) / len(cluster_river)
        local_all_coherence_river.append(r_river)

        cluster_financial = [local_phases['bank'], local_phases['loan'], local_phases['money']]
        r_financial = np.abs(np.sum(np.exp(1j * np.array(cluster_financial)))) / len(cluster_financial)
        local_all_coherence_financial.append(r_financial)

        # Calculate and store the phase difference for the new analysis
        phase_diff_river = local_phases['bank'] - local_phases['water']
        local_all_phase_diff_river.append(phase_diff_river)

        phase_diff_financial = local_phases['bank'] - local_phases['money']
        local_all_phase_diff_financial.append(phase_diff_financial)
        
    return local_all_phases, local_all_couplings, local_all_coherence_river, local_all_coherence_financial, local_couplings, local_phases, local_all_phase_diff_river, local_all_phase_diff_financial


# --- Simulation with Balanced Omegas ---
# Run Training
all_phases_1_b, all_couplings_1_b, all_coherence_river_1_b, all_coherence_financial_1_b, final_couplings_1_b, final_phases_1_b, phase_diff_river_1, phase_diff_financial_1 = \
    run_simulation(phases, coupling_strengths, 0, training_duration_river, use_attention=True, omega_set=balanced_omega)
all_phases_2_b, all_couplings_2_b, all_coherence_river_2_b, all_coherence_financial_2_b, final_couplings_2_b, final_phases_2_b, phase_diff_river_2, phase_diff_financial_2 = \
    run_simulation(final_phases_1_b, final_couplings_1_b, training_duration_river, training_duration_river + training_duration_financial, use_attention=True, omega_set=balanced_omega)

# Run Probing
probe_phases = {key: np.random.uniform(0, 2*np.pi) for key in balanced_omega}
probe_phases['bank'] += 0.1 
all_phases_3_b, all_couplings_3_b, all_coherence_river_3_b, all_coherence_financial_3_b, final_couplings_3_b, final_phases_3_b, phase_diff_river_3, phase_diff_financial_3 = \
    run_simulation(probe_phases, final_couplings_2_b, training_duration_river + training_duration_financial + pause_duration, total_time, use_attention=False, omega_set=balanced_omega)

# --- Concatenate all phases of the balanced simulation for animation ---
all_phases_final = {}
for key in balanced_omega.keys():
    all_phases_final[key] = all_phases_1_b[key] + all_phases_2_b[key] + all_phases_3_b[key]

all_couplings_final = {}
for key in coupling_strengths.keys():
    all_couplings_final[key] = all_couplings_1_b[key] + all_couplings_2_b[key] + all_couplings_3_b[key]

all_coherence_river_final = all_coherence_river_1_b + all_coherence_river_2_b + all_coherence_river_3_b
all_coherence_financial_final = all_coherence_financial_1_b + all_coherence_financial_2_b + all_coherence_financial_3_b

all_phase_diff_river_final = phase_diff_river_1 + phase_diff_river_2 + phase_diff_river_3
all_phase_diff_financial_final = phase_diff_financial_1 + phase_diff_financial_2 + phase_diff_financial_3

# --- Fourier Analysis on the Probing Phase ---
# We analyze a section of the probing phase after the network has settled
settling_time_start = context_switch_time_2 + 2.0
settling_time_end = total_time

start_index_fft = int(settling_time_start / dt)
end_index_fft = int(settling_time_end / dt)

# Kuramoto order parameter for river cluster during a settled portion of probing
R_river = all_coherence_river_final[start_index_fft:end_index_fft]
# Kuramoto order parameter for financial cluster during a settled portion of probing
R_financial = all_coherence_financial_final[start_index_fft:end_index_fft]

# Phase differences for new FFT plots
phase_diff_river_fft_data = np.sin(all_phase_diff_river_final[start_index_fft:end_index_fft])
phase_diff_financial_fft_data = np.sin(all_phase_diff_financial_final[start_index_fft:end_index_fft])

N = len(R_river)
T = settling_time_end - settling_time_start

# FFT for river cluster coherence
yf_river = fft(R_river)
xf_river = fftfreq(N, dt)[:N//2]
yf_river_magnitude = 2.0/N * np.abs(yf_river[0:N//2])

# FFT for financial cluster coherence
yf_financial = fft(R_financial)
xf_financial = fftfreq(N, dt)[:N//2]
yf_financial_magnitude = 2.0/N * np.abs(yf_financial[0:N//2])

# NEW FFT for phase differences
yf_diff_river = fft(phase_diff_river_fft_data)
yf_diff_river_magnitude = 2.0/N * np.abs(yf_diff_river[0:N//2])

yf_diff_financial = fft(phase_diff_financial_fft_data)
yf_diff_financial_magnitude = 2.0/N * np.abs(yf_diff_financial[0:N//2])

# --- Set up the figure for animation and plots ---
fig = plt.figure(figsize=(15, 12))
gs = GridSpec(3, 2, figure=fig)

# --- Plot 1: Oscillator Network ---
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_title("Oscillator Network (Balanced Omegas)")
ax0.axis('off')
ax0.set_aspect('equal')

# --- Plot 2: Oscillator Phases ---
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_title("Oscillator Phases Over Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Oscillator State (sin(θ))")
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True)
ax1.axvline(context_switch_time_1, color='k', linestyle='--', alpha=0.5, label='Context Switch 1')
ax1.axvline(context_switch_time_2, color='k', linestyle='--', alpha=0.5, label='Probe Start')
ax1.axvline(settling_time_start, color='g', linestyle='--', alpha=0.5, label='FFT Start')
ax1.legend(loc='lower right')

# --- Plot 3: Coherence ---
ax2 = fig.add_subplot(gs[2, 0])
ax2.set_title("Cluster Coherence (Kuramoto Order Parameter)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Coherence (r)")
ax2.set_ylim(0, 1.0)
ax2.grid(True)
ax2.axvline(context_switch_time_1, color='k', linestyle='--', alpha=0.5)
ax2.axvline(context_switch_time_2, color='k', linestyle='--', alpha=0.5)
ax2.axvline(settling_time_start, color='g', linestyle='--', alpha=0.5)

# --- Plot 4: Fourier Transform of River Cluster Coherence ---
ax3 = fig.add_subplot(gs[0, 1])
ax3.set_title("FFT of River Bank Coherence")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Magnitude")
ax3.grid(True)
ax3.plot(xf_river, yf_river_magnitude, 'b-')
# ax3.axvline(x=7.0, color='b', linestyle='--', label='Expected Peak at 7 Hz')
ax3.legend(loc='upper right')
ax3.set_xlim(0, 20)

# --- Plot 5: Fourier Transform of Financial Cluster Coherence ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title("FFT of Financial Coherence")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Magnitude")
ax4.grid(True)
ax4.plot(xf_financial, yf_financial_magnitude, 'r-')
# ax4.axvline(x=3.0, color='r', linestyle='--', label='Expected Peak at 3 Hz')
ax4.legend(loc='upper right')
ax4.set_xlim(0, 20)

# --- Plot 6: NEW - FFT of Phase Differences ---
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_title("FFT of Phase Differences with 'bank' node")
ax5.set_xlabel("Frequency (Hz)")
ax5.set_ylabel("Magnitude")
ax5.grid(True)
ax5.plot(xf_river, yf_diff_river_magnitude, 'b-', label="River Bank vs Bank")
ax5.plot(xf_financial, yf_diff_financial_magnitude, 'r-', label="Financial vs Bank")
# ax5.axvline(x=3.0, color='r', linestyle='--', alpha=0.7)
# ax5.axvline(x=7.0, color='b', linestyle='--', alpha=0.7)
ax5.set_xlim(0, 20)
ax5.legend(loc='upper right')


# --- Plot initialization for the animation ---
node_positions = {
    'bank': (0, 0), 'loan': (-0.5, 0.7), 'money': (0.5, 0.7),
    'water': (-0.5, -0.7), 'swim': (0.5, -0.7),
}
node_base_colors = {
    'bank': 'red', 'loan': 'purple', 'water': 'blue',
    'swim': 'cyan', 'money': 'green',
}

ax0.set_xlim(-1.5, 1.5)
ax0.set_ylim(-1.5, 1.5)

nodes = {
    name: ax0.add_patch(Circle(pos, 0.1, color=node_base_colors[name], zorder=2))
    for name, pos in node_positions.items()
}
node_texts = {
    name: ax0.text(pos[0], pos[1] + 0.15, name, ha='center', va='center', fontsize=8)
    for name, pos in node_positions.items()
}

connections = {}
for (n1, n2) in all_couplings_final.keys():
    if n1 in node_positions and n2 in node_positions:
        pos1 = node_positions[n1]
        pos2 = node_positions[n2]
        connections[(n1, n2)] = ax0.add_patch(ConnectionPatch(
            xyA=pos1, xyB=pos2, coordsA='data', coordsB='data',
            color='black', alpha=0.5, zorder=1, linewidth=0
        ))

line_plots = {name: ax1.plot([], [], label=name, color=node_base_colors[name])[0] for name in balanced_omega.keys()}

line_coherence_river, = ax2.plot([], [], 'b-', label='River Bank Cluster')
line_coherence_financial, = ax2.plot([], [], 'r-', label='Financial Cluster')
ax2.legend(loc='upper right')

# --- The Animation Update Function ---
def update(frame):
    current_time_slice = t[:frame+1]
    
    for name, line in line_plots.items():
        line.set_data(current_time_slice, np.sin(all_phases_final[name][:frame+1]))
        
    line_coherence_river.set_data(current_time_slice, all_coherence_river_final[:frame+1])
    line_coherence_financial.set_data(current_time_slice, all_coherence_financial_final[:frame+1])
    
    coupling_index = frame
    
    for name, node_patch in nodes.items():
        phase_sin = np.sin(all_phases_final[name][coupling_index])
        bright_factor = (phase_sin + 1) / 2
        original_color = np.array(colors.to_rgb(node_base_colors[name]))
        new_color = original_color * (0.5 + 0.5 * bright_factor)
        node_patch.set_facecolor(new_color)
    
    for (n1, n2), conn in connections.items():
        strength = all_couplings_final.get((n1, n2), [0] * time_steps)[coupling_index]
        conn.set_linewidth(strength * 5)
        conn.set_alpha(min(1.0, strength / 0.5))

    return list(line_plots.values()) + [line_coherence_river, line_coherence_financial] + list(connections.values()) + list(nodes.values())

# Create the animation
ani = FuncAnimation(fig, update, frames=time_steps, interval=25, blit=False)

plt.tight_layout()
plt.show()

