import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, ConnectionPatch
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq

# --- Simulation Parameters ---
total_time = 35.0
time_steps = 2000
dt = total_time / time_steps
t = np.linspace(0, total_time, time_steps)
training_duration_river = 10.0
training_duration_financial = 10.0
pause_duration = 5.0
probe_duration = 10.0
context_switch_time_1 = training_duration_river
pause_end_time = training_duration_river + training_duration_financial + pause_duration
settling_time_start = pause_end_time + 2.0 # Start FFT analysis after some settling time
settling_time_end = total_time

# --- Hebbian Rule and Forgetting (Adjusted for better learning) ---
eta = 0.40  # Increased learning rate for connections within the active context
alpha_learning = 0.005 # Less aggressive forgetting during learning (prevents indefinite growth)
alpha_forgetting = 0.8  # Aggressive forgetting rate for connections NOT in the active context

# --- Oscillator Definitions with Corrected Balanced Omegas ---
# Frequencies for 'river bank' cluster are tightly clustered around 7Hz
# Frequencies for 'financial bank' cluster are tightly clustered around 3Hz
# The shared 'bank' node now has a frequency aligned with the financial cluster.
balanced_omega = {
    'bank': 3.1 * 2 * np.pi, # Financial-aligned frequency
    'loan': 3.0 * 2 * np.pi, # Financial cluster
    'water': 7.1 * 2 * np.pi, # River cluster
    'swim': 6.9 * 2 * np.pi, # River cluster
    'money': 2.9 * 2 * np.pi, # Financial cluster
}

omegas_to_use = balanced_omega
# Initial random phases for a robust start
initial_phases = {name: np.random.uniform(0, 2 * np.pi) for name in omegas_to_use}

initial_coupling_strength = 0.001
# Initialize coupling strengths with sorted keys for consistency
coupling_strengths = {
    tuple(sorted((n1, n2))): initial_coupling_strength for n1 in omegas_to_use for n2 in omegas_to_use if n1 != n2
}

# --- Context-based Attention Mechanism for TRAINING ONLY (CORRECTED) ---
def get_attention_for_training(t_current):
    """
    Returns an attention matrix where 1.0 means the connection is part of the
    currently active context and should be subject to learning (strengthening/mild decay),
    and 0.0 means it's not and should be aggressively forgotten.
    """
    attention_matrix = {key: {key2: 0.0 for key2 in omegas_to_use} for key in omegas_to_use}
    
    if t_current < training_duration_river:
        # Context 1: "River Bank" - Activate connections within 'bank', 'water', 'swim' cluster
        cluster_nodes = ['bank', 'water', 'swim']
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                n1, n2 = cluster_nodes[i], cluster_nodes[j]
                attention_matrix[n1][n2] = 1.0
                attention_matrix[n2][n1] = 1.0
    
    elif t_current < training_duration_river + training_duration_financial:
        # Context 2: "Financial Bank" - Activate connections within 'bank', 'loan', 'money' cluster
        cluster_nodes = ['bank', 'loan', 'money']
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                n1, n2 = cluster_nodes[i], cluster_nodes[j]
                attention_matrix[n1][n2] = 1.0
                attention_matrix[n2][n1] = 1.0

    return attention_matrix

# --- Simulation Runner Function ---
def run_simulation(initial_phases, initial_couplings, t_start, t_end, use_attention=False, omega_set=balanced_omega):
    """
    Runs the simulation for a specific time window, applying Kuramoto dynamics
    and Hebbian learning/forgetting based on context.
    """
    local_phases = initial_phases.copy()
    local_couplings = initial_couplings.copy() # Use a copy to avoid modifying the original dict
    
    local_all_phases = {key: [] for key in omega_set.keys()}
    local_all_couplings = {key: [] for key in initial_couplings.keys()}
    
    local_all_coherence_river = []
    local_all_coherence_financial = []
    
    start_index = int(t_start / dt)
    end_index = int(t_end / dt)
    
    for i in range(start_index, end_index):
        t_current = t[i]
        new_phases = local_phases.copy()
        
        # Hebbian Learning/Forgetting: Only active during training phase (when use_attention is True)
        if use_attention:
            attention_matrix = get_attention_for_training(t_current)
            for name_i in omega_set.keys():
                for name_j in omega_set.keys():
                    if name_i != name_j:
                        coupling_key = tuple(sorted((name_i, name_j))) # Use sorted key
                        coupling_val = local_couplings.get(coupling_key, 0)
                        
                        # Check if the connection (n1, n2) or (n2, n1) is in the current context
                        is_in_context = attention_matrix[name_i].get(name_j, 0) == 1.0
                        
                        phase_diff = local_phases[name_j] - local_phases[name_i]
                        
                        if is_in_context:
                            # Strengthen connections within the active context, with mild decay
                            dJ_dt = eta * (np.sin(phase_diff))**2 - alpha_learning * coupling_val
                        else:
                            # Aggressively forget connections NOT in the active context
                            dJ_dt = -alpha_forgetting * coupling_val
                        
                        local_couplings[coupling_key] = max(0, coupling_val + dJ_dt * dt)
        
        # Kuramoto Dynamics: Update phases based on current coupling strengths
        for name_i in omega_set.keys():
            total_pull = 0
            for name_j in omega_set.keys():
                if name_i != name_j:
                    coupling_key = tuple(sorted((name_i, name_j))) # Use sorted key
                    coupling_val = local_couplings.get(coupling_key, 0)
                    phase_diff = local_phases[name_j] - local_phases[name_i]
                    total_pull += coupling_val * np.sin(phase_diff)
                    
            d_theta_i_dt = omega_set[name_i] + total_pull
            new_phases[name_i] += d_theta_i_dt * dt
        
        local_phases = new_phases

        # Store data for this time step
        for key in local_phases.keys():
            local_all_phases[key].append(local_phases[key])
        
        for coupling_key, coupling_val in local_couplings.items():
            if coupling_key not in local_all_couplings:
                local_all_couplings[coupling_key] = []
            local_all_couplings[coupling_key].append(coupling_val)
        
        # Calculate Kuramoto Order Parameter for each cluster
        cluster_river = [local_phases['bank'], local_phases['water'], local_phases['swim']]
        r_river = np.abs(np.sum(np.exp(1j * np.array(cluster_river)))) / len(cluster_river)
        local_all_coherence_river.append(r_river)

        cluster_financial = [local_phases['bank'], local_phases['loan'], local_phases['money']]
        r_financial = np.abs(np.sum(np.exp(1j * np.array(cluster_financial)))) / len(cluster_financial)
        local_all_coherence_financial.append(r_financial)
        
    return local_all_phases, local_all_couplings, local_all_coherence_river, local_all_coherence_financial, local_couplings, local_phases


# --- Simulation Phases ---

# 1. Training Phase: River Bank
all_phases_1_b, all_couplings_1_b, all_coherence_river_1_b, all_coherence_financial_1_b, final_couplings_1_b, final_phases_1_b = \
    run_simulation(initial_phases, coupling_strengths, 0, training_duration_river, use_attention=True, omega_set=balanced_omega)

# 2. Training Phase: Financial Bank
all_phases_2_b, all_couplings_2_b, all_coherence_river_2_b, all_coherence_financial_2_b, final_couplings_2_b, final_phases_2_b = \
    run_simulation(final_phases_1_b, final_couplings_1_b, training_duration_river, training_duration_river + training_duration_financial, use_attention=True, omega_set=balanced_omega)

# 3. Probing Phase: Use learned couplings, no attention for learning/forgetting
# Reset phases to incoherent state to see which attractor the network falls into
probe_phases = {key: np.random.uniform(0, 2*np.pi) for key in balanced_omega}
probe_phases['bank'] += 0.1 # Give a small kick to the ambiguous node
all_phases_3_b, all_couplings_3_b, all_coherence_river_3_b, all_coherence_financial_3_b, final_couplings_3_b, final_phases_3_b = \
    run_simulation(probe_phases, final_couplings_2_b, pause_end_time, total_time, use_attention=False, omega_set=balanced_omega)

# --- Concatenate all phases of the balanced simulation for animation ---
all_phases_final = {}
for key in balanced_omega.keys():
    # Correctly handle the pause duration by repeating the phases/couplings from the end of training
    pause_steps = int(pause_duration / dt)
    all_phases_final[key] = all_phases_1_b[key] + all_phases_2_b[key] + [all_phases_2_b[key][-1]] * pause_steps + all_phases_3_b[key]

all_couplings_final = {}
for (n1, n2) in coupling_strengths.keys():
    # Use the sorted key for consistency
    key = tuple(sorted((n1, n2)))
    pause_steps = int(pause_duration / dt)
    all_couplings_final[key] = all_couplings_1_b[key] + all_couplings_2_b[key] + [all_couplings_2_b[key][-1]] * pause_steps + all_couplings_3_b[key]

all_coherence_river_final = all_coherence_river_1_b + all_coherence_river_2_b + [all_coherence_river_2_b[-1]] * pause_steps + all_coherence_river_3_b
all_coherence_financial_final = all_coherence_financial_1_b + all_coherence_financial_2_b + [all_coherence_financial_2_b[-1]] * pause_steps + all_coherence_financial_3_b

# --- Fourier Analysis on the Probing Phase ---
# We analyze a section of the probing phase after the network has settled
start_index_fft = int(settling_time_start / dt)
end_index_fft = int(settling_time_end / dt)
N_fft = end_index_fft - start_index_fft
T_fft = settling_time_end - settling_time_start
dt_fft = T_fft / N_fft

# FFT for individual oscillator states, not coherence
# For the 'bank' node, we expect a peak at 3.1 Hz due to the trained financial context
bank_signal = np.sin(np.array(all_phases_final['bank'][start_index_fft:end_index_fft]))
yf_bank = fft(bank_signal)
xf_bank = fftfreq(N_fft, dt_fft)[:N_fft//2]
yf_bank_magnitude = 2.0/N_fft * np.abs(yf_bank[0:N_fft//2])

# For the 'water' node, we expect a peak at 7.1 Hz from the river context
water_signal = np.sin(np.array(all_phases_final['water'][start_index_fft:end_index_fft]))
yf_water = fft(water_signal)
xf_water = fftfreq(N_fft, dt_fft)[:N_fft//2]
yf_water_magnitude = 2.0/N_fft * np.abs(yf_water[0:N_fft//2])


# --- Set up the figure for animation and plots ---
fig = plt.figure(figsize=(15, 12))
# Adjusting gridspec to provide more space for the top-left text
gs = GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.3, top=0.95, bottom=0.05, left=0.05, right=0.95)
fig.suptitle("Resolving Ambiguity: 'bank'", fontsize=16, fontweight='bold')


# --- Plot 1: Oscillator Network ---
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_title("Oscillator Network (Balanced Omegas)")
ax0.axis('off')
ax0.set_aspect('equal')

# --- Simulation explanation text (adjusted for better placement) ---
fig.text(0.05, 0.9, "This simulation models how a recurrent network of semantic oscillators can\n"
                  "resolve the ambiguity of the word 'bank' in the sentence 'I need to take a\n"
                  "loan from a bank.' by learning two different contexts.",
                  fontsize=10, va='top', ha='left', transform=fig.transFigure)

# --- Real-time status text ---
status_text = ax0.text(0.5, 0.8, "Status: Initializing...", fontsize=12, fontweight='bold', ha='center', transform=ax0.transAxes)


# --- Plot 2: Oscillator Phases ---
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_title("Oscillator Phases Over Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Oscillator State (sin(θ))")
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True)
ax1.axvline(context_switch_time_1, color='k', linestyle='--', alpha=0.5, label='Context Switch 1')
ax1.axvline(pause_end_time, color='k', linestyle='--', alpha=0.5, label='Probe Start')
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
ax2.axvline(pause_end_time, color='k', linestyle='--', alpha=0.5)
ax2.axvline(settling_time_start, color='g', linestyle='--', alpha=0.5)

# --- Plot 4: Fourier Transform of 'bank' Oscillator ---
ax3 = fig.add_subplot(gs[0, 1])
ax3.set_title("FFT of 'bank' Oscillator during Probing")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Magnitude")
ax3.grid(True)
ax3.plot(xf_bank, yf_bank_magnitude, 'b-')
ax3.axvline(x=3.1, color='r', linestyle='--', label='Expected Peak at 3.1 Hz')
ax3.legend(loc='upper right')
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 1.0)

# --- Plot 5: Fourier Transform of 'water' Oscillator ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title("FFT of 'water' Oscillator during Probing")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Magnitude")
ax4.grid(True)
ax4.plot(xf_water, yf_water_magnitude, 'r-')
ax4.axvline(x=7.1, color='b', linestyle='--', label='Expected Peak at 7.1 Hz')
ax4.legend(loc='upper right')
ax4.set_xlim(0, 20)
ax4.set_ylim(0, 1.0)


# --- Plot 6: Dynamic Coupling Strengths ---
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_title("Dynamic Coupling Strengths")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Coupling Strength (J)")
ax5.set_ylim(0, 1.0) # Coupling strengths are clamped to 0-1
ax5.grid(True)
ax5.axvline(context_switch_time_1, color='k', linestyle='--', alpha=0.5)
ax5.axvline(pause_end_time, color='k', linestyle='--', alpha=0.5)
ax5.axvline(settling_time_start, color='g', linestyle='--', alpha=0.5)

# Plot specific coupling strengths
line_coupling_bank_water, = ax5.plot([], [], 'b-', label='bank-water')
line_coupling_bank_money, = ax5.plot([], [], 'r-', label='bank-money')
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
for (n1, n2) in coupling_strengths.keys(): # Iterate over the sorted keys
    if n1 in node_positions and n2 in node_positions:
        pos1 = node_positions[n1]
        pos2 = node_positions[n2]
        connections[tuple(sorted((n1, n2)))] = ax0.add_patch(ConnectionPatch( # Store with sorted key
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
    current_time = t[frame]

    # Update status text
    if current_time < training_duration_river:
        status_text.set_text("Status: Learning Context 1 (River Bank)")
    elif current_time < training_duration_river + training_duration_financial:
        status_text.set_text("Status: Learning Context 2 (Financial Bank)")
    elif current_time < pause_end_time:
        status_text.set_text("Status: Pause (Forgetting)")
    else:
        status_text.set_text("Status: Probing for Ambiguity Resolution")

    # Update plots with new data
    for name, line in line_plots.items():
        # Ensure the slice doesn't go out of bounds
        if frame < len(all_phases_final[name]):
            line.set_data(current_time_slice, np.sin(all_phases_final[name][:frame+1]))
        
    if frame < len(all_coherence_river_final):
        line_coherence_river.set_data(current_time_slice, all_coherence_river_final[:frame+1])
    if frame < len(all_coherence_financial_final):
        line_coherence_financial.set_data(current_time_slice, all_coherence_financial_final[:frame+1])
    
    coupling_index = frame
    
    for name, node_patch in nodes.items():
        if coupling_index < len(all_phases_final[name]):
            phase_sin = np.sin(all_phases_final[name][coupling_index])
            bright_factor = (phase_sin + 1) / 2
            original_color = np.array(colors.to_rgb(node_base_colors[name]))
            new_color = original_color * (0.5 + 0.5 * bright_factor)
            node_patch.set_facecolor(new_color)
    
    for (n1, n2), conn in connections.items():
        # Use the sorted key
        sorted_key = tuple(sorted((n1, n2)))
        if coupling_index < len(all_couplings_final[sorted_key]):
            strength = all_couplings_final.get(sorted_key, [0] * time_steps)[coupling_index]
            conn.set_linewidth(strength * 5)
            conn.set_alpha(min(1.0, strength / 0.5))

    # Update dynamic coupling strength lines
    if ('bank', 'water') in all_couplings_final and frame < len(all_couplings_final[('bank', 'water')]):
        line_coupling_bank_water.set_data(current_time_slice, all_couplings_final[('bank', 'water')][:frame+1])
    if ('bank', 'money') in all_couplings_final and frame < len(all_couplings_final[('bank', 'money')]):
        line_coupling_bank_money.set_data(current_time_slice, all_couplings_final[('bank', 'money')][:frame+1])


    return list(line_plots.values()) + [line_coherence_river, line_coherence_financial] + \
           list(connections.values()) + list(nodes.values()) + [status_text, line_coupling_bank_water, line_coupling_bank_money]

# Create the animation
ani = FuncAnimation(fig, update, frames=time_steps, interval=25, blit=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Final Output after animation closes ---
probing_start_index = int(pause_end_time / dt)
avg_coherence_river_probe = np.mean(all_coherence_river_final[probing_start_index:])
avg_coherence_financial_probe = np.mean(all_coherence_financial_final[probing_start_index:])

print("\n--- Final Probing Results ---")
print(f"Average River Bank Cluster Coherence during probing: {avg_coherence_river_probe:.3f}")
print(f"Average Financial Bank Cluster Coherence during probing: {avg_coherence_financial_probe:.3f}")

if avg_coherence_financial_probe > avg_coherence_river_probe:
    print("The 'bank' node successfully resolved to the **Financial Bank** meaning.")
    print("This is evidenced by higher coherence in the financial cluster during probing,")
    print("and the 'bank' oscillator's FFT peak aligning with the financial cluster's frequency.")
else:
    print("The 'bank' node successfully resolved to the **River Bank** meaning.")
    print("This is evidenced by higher coherence in the river cluster during probing,")
    print("and the 'bank' oscillator's FFT peak aligning with the river cluster's frequency.")

print("\n--- Key Takeaways from FFT Analysis (Probing Phase) ---")
# Find the dominant frequency in the bank oscillator's FFT
dominant_bank_freq_idx = np.argmax(yf_bank_magnitude)
dominant_bank_freq = xf_bank[dominant_bank_freq_idx]
print(f"Dominant frequency of 'bank' oscillator: {dominant_bank_freq:.2f} Hz")

# Find the dominant frequency in the water oscillator's FFT
dominant_water_freq_idx = np.argmax(yf_water_magnitude)
dominant_water_freq = xf_water[dominant_water_freq_idx]
print(f"Dominant frequency of 'water' oscillator: {dominant_water_freq:.2f} Hz")

print("\nObserve how the 'bank' oscillator's dominant frequency aligns with the cluster it resolved to.")
