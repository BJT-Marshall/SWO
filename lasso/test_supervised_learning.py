# Test Lasso Supervised Learning: Small System

from GPSKet.operator.hamiltonian import J1J2
import netket as nk
from lasso import lasso_wf_optimisation, system_setup, apply_model
from heatmaps import heatmap_from_array
from sampling import basis_measurement_sampling as wf_sample
import jax.numpy as jnp
import matplotlib.pyplot as plt

import os

# System and test setup

test_ind = 2

Lx = 4
Ly = 2
M=10
training_fraction = 1
iters = 50
alpha = 5*10**(-9)
scaling = True
heatmaps = True

ha = J1J2.get_J1_J2_Hamiltonian(Lx=Lx, Ly = Ly, J2=0, sign_rule=[True,False], on_the_fly_en=False)
g = nk.graph.Grid([Lx,Ly], pbc = True)
vs_r,vs_i = system_setup(hilbert = ha.hilbert, graph = g, M=M, seed = 1, smp_seed = 1)

path = 'Test'+str(test_ind)
if not os.path.exists(path):
    os.makedirs(path)

with open(path+'/SetupTest'+str(test_ind),'w') as f:
    f.write("Lx = "+str(Lx)+"\n")
    f.write("Ly = "+str(Ly)+"\n")
    f.write("M = "+str(M)+"\n")
    f.write("training_fraction = "+str(training_fraction)+", ("+str(int(training_fraction*ha.hilbert.n_states))+" data points)""\n")
    f.write("Iters = "+str(iters)+"\n")
    f.write("Alpha = "+str(alpha)+"\n")
    f.write("Scaling = "+str(scaling)+"\n")


# Data generation:

e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
amps = state.flatten()
d = len(amps)

log_amps_R = jnp.log(jnp.abs(amps))
log_amps_I = jnp.angle(amps)
log_amps = [log_amps_R[i]+1j*log_amps_I[i] for i in range(d)]

# Config/label generation

print("pre configs")
configs = ha.hilbert.all_states()
print("configs generated")

# Sampling training data

indices, _ = wf_sample(amps, d*training_fraction,True)

training_data = jnp.array([log_amps[i] for i in indices])
training_labels = jnp.array([configs[i] for i in indices])

print("training data sampled")


# Supervised learning

pred_r, pred_i, phase_shift = lasso_wf_optimisation(vs_r,vs_i,training_data,training_labels,iters,alpha,scaling)

print("learning finished")


def write_epsilon_to_file(epsilon, test_ind, name):
    with open('Test'+str(test_ind)+'/Epsilon_'+str(name),'w') as f:
        f.write(str(epsilon.shape[0])+" "+str(epsilon.shape[1])+" "+str(epsilon.shape[2])+"\n")
        epsilon = epsilon.flatten()
        for i in range(len(epsilon)):
            f.write(str(epsilon[i])+"\n")
        f.close()

write_epsilon_to_file(pred_r, test_ind, "pred_r")
write_epsilon_to_file(pred_i, test_ind, "pred_i")


print("epsilons written to file")

if heatmaps:
    heatmap_from_array(pred_r, 'Test'+str(test_ind)+'/Epsilon_pred_r_heatmap.png')
    heatmap_from_array(pred_i, 'Test'+str(test_ind)+'/Epsilon_pred_i_heatmap.png')
    print("heatmaps generated")

# Generate predicted log amplitudes for the set of training data and the full wavefunction

training_data_fit = apply_model(vs_r,vs_i,pred_r,pred_i,phase_shift,training_labels)
print("training fit generated")
full_data_fit = apply_model(vs_r,vs_i,pred_r,pred_i,phase_shift,configs)
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
o1=overlap(norm_training_data,norm_training_data_fit)
print("Training Data Overlap:", o1)

plt.plot([i for i in range(len(training_data_amps))],norm_training_data, color = 'r', label = 'data')
plt.plot([i for i in range(len(training_data_amps))],norm_training_data_fit, color = 'b', label = 'fit')
plt.title("Overlap = "+str(o1))
plt.legend()
plt.savefig('Test'+str(test_ind)+'/DataTest'+str(test_ind))

plt.clf()

o2=overlap(amps,full_data_fit)
print("Full Data Overlap:", o2)

plt.plot([i for i in range(d)],amps, color = 'r', label = 'data')
plt.plot([i for i in range(d)],full_data_fit, color = 'b', label = 'fit')
plt.title("Overlap = "+str(o2))
plt.legend()
plt.savefig('Test'+str(test_ind)+'/DataTestFull'+str(test_ind))

plt.clf()










