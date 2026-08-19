# Test Lasso Supervised Learning: Small System

from GPSKet.operator.hamiltonian import J1J2
import netket as nk
from lasso import lasso_wf_optimisation, system_setup, apply_model
from sampling import basis_measurement_sampling as wf_sample
import jax.numpy as jnp
import matplotlib.pyplot as plt

# System setup

Lx = 4
Ly = 4

ha = J1J2.get_J1_J2_Hamiltonian(Lx=Lx, Ly = Ly, J2=0, sign_rule=[True,False], on_the_fly_en=False)
g = nk.graph.Grid([Lx,Ly], pbc = True)
vs_r,vs_i = system_setup(hilbert = ha.hilbert, graph = g, M=100, seed = 1, smp_seed = 1)

# Data generation:

e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
amps = state.flatten()
d = len(amps)

log_amps_R = jnp.log(jnp.abs(amps))
log_amps_I = jnp.angle(amps)
log_amps = [log_amps_R[i]+1j*log_amps_I[i] for i in range(d)]

# Config/label generation

print("pre configs")
configs = jnp.array(ha.hilbert.states_to_local_indices(ha.hilbert.all_states()))
print("configs generated")

# Sampling training data

training_fraction = 0.1

indices, _ = wf_sample(amps, d*training_fraction,True)

training_data = jnp.array([log_amps[i] for i in indices])
training_labels = jnp.array([configs[i] for i in indices])

print("training data sampled")


# Supervised learning

pred_r, pred_i, phase_shift = lasso_wf_optimisation(vs_r,vs_i,training_data,training_labels,15,10**-7,True)

print("learning finished")

# Generate predicted log amplitudes for the set of training data and the full wavefunction

training_data_fit = apply_model(vs_r,vs_i,pred_r.epsilon,pred_i.epsilon,phase_shift,training_labels)
print("training fit generated")
full_data_fit = apply_model(vs_r,vs_i,pred_r.epsilon,pred_i.epsilon,phase_shift,configs)
print("full fit generated")

# Testing

def overlap(array1, array2):

    array1 = jnp.array(array1)/jnp.linalg.norm(array1)
    array1 = array1.conj().T
    array2 = jnp.array(array2)/jnp.linalg.norm(array2)

    o = abs(array1.dot(array2))

    return o

# Generate a normalised distribution out of the training data for visualisation
training_data_amps = jnp.exp(training_data)/jnp.linalg.norm(jnp.exp(training_data))
norm_training_data = training_data_amps/jnp.linalg.norm(training_data_amps)

# Generate a normalised distribution out of the training data fit for visualisation
norm_training_data_fit = training_data_fit/jnp.linalg.norm(training_data_fit)

# Generate a prediction for the full normalised wavefunction
norm_full_fir = full_data_fit/jnp.linalg.norm(full_data_fit)

# Plotting and print overlaps

print("Training Data Overlap:", overlap(norm_training_data,norm_training_data_fit))

plt.plot([i for i in range(len(training_data_amps))],norm_training_data, color = 'r', label = 'data')
plt.plot([i for i in range(len(training_data_amps))],norm_training_data_fit, color = 'b', label = 'fit')
plt.legend()
plt.savefig('training_data10')

plt.clf()

print("Full Data Overlap:", overlap(amps,full_data_fit))

plt.plot([i for i in range(d)],amps, color = 'r', label = 'data')
plt.plot([i for i in range(d)],full_data_fit, color = 'b', label = 'fit')
plt.legend()
plt.savefig('full_data10')

plt.clf()










