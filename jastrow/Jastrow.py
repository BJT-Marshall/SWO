#Jastrow Model Generation for Small Test Systems

import netket as nk
import jax.numpy as jnp
from GPSKet.operator.hamiltonian import J1J2
import numpy as np
import matplotlib.pyplot as plt
import optax
import jax
import os


dataset_ind = 1 #REMEMBER TO GIVE A NEW INDEX TO NOT OVERWRITE DATA

# System setup
system_dims = [4,2]
J2 = 0

if system_dims[0] == 1:
    Lx = system_dims[1]
    Ly = None
elif system_dims[1] == 1:
    Lx = system_dims[0]
    Ly = None
else:
    Lx = system_dims[0]
    Ly = system_dims[1]


N = system_dims[0]*system_dims[1]



hilbert = nk.hilbert.Spin(s=0.5, N=N)
ha = J1J2.get_J1_J2_Hamiltonian(Lx=Lx, Ly = Ly, J2=J2, sign_rule=[True, False], on_the_fly_en=False)
model = nk.models.Jastrow()


e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
amps = state.flatten()

# Enforce global phase convention by making the largest amplitude positive

max_idx = np.argmax(np.abs(amps))
if amps[max_idx] < 0:
    amps *= -1


sampler = nk.sampler.MetropolisLocal(hilbert)

# Full-summation variational state
vs = nk.vqs.FullSumState(
    hilbert=hilbert,
    model=model,
)

# Initialize parameters
vs.init_parameters()

target = amps.astype(np.complex128)

# Access the parameter PyTree
params = vs.parameters

# ----------------------------------Fitting loop to fit the Jastrow model to the target amplitudes here------------------------------------------

optimizer = optax.adam(learning_rate=1e-2)
# initialise optimizer state

params = vs.parameters

opt_state = optimizer.init(params)

def loss_fn(params):
    # log amplitudes from Jastrow ansatz
    logpsi = vs.model.apply({"params": params},ha.hilbert.all_states())
    # normalized variational wavefunction
    psi = jnp.exp(logpsi)
    psi = psi/jnp.linalg.norm(psi)
    overlap = jnp.vdot(amps, psi)
    fidelity = jnp.abs(overlap)**2
    
    return 1 - fidelity

loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
n_iter = 5000
for step in range(n_iter):
    loss, grads = loss_and_grad(params)
    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        params
        )
params = optax.apply_updates(params, updates)

print("Final Loss:", loss)

# put the updated parameters back into the variational state
vs.parameters = params


#-----------------------------------------------------------------------------------------------------------------------------------------------

pred_amps =vs.model.apply({"params":params}, ha.hilbert.all_states())
pred_amps /=np.linalg.norm(pred_amps)

full_configs = [[(n >> i) & 1 for i in range(7, -1, -1)] for n in range(2**(N))]
pred_amps_full=vs.model.apply({"params":params}, ha.hilbert.local_indices_to_states(full_configs))
pred_amps_full /=np.linalg.norm(pred_amps_full)

# Enforce global phase convention by making the largest amplitude positive
max_idx = np.argmax(np.abs(pred_amps))
if pred_amps[max_idx] < 0:
    pred_amps *= -1

#Save the jastrow model parameters to a file

path = 'jastrow/data/dataset'+str(dataset_ind)
if not os.path.exists(path):
    os.makedirs(path)

with open("jastrow/data/dataset"+str(dataset_ind)+"/SystemData", "w") as f:
    if Lx == None:
        Lx_ = 1
    else:
        Lx_ = Lx
    f.write("Lx = "+str(Lx_)+"\n")
    if Ly == None:
        Ly_ = 1
    else:
        Ly_ = Ly
    f.write("Ly = "+str(Ly_)+"\n")
    f.write("J2 = "+str(J2)+"\n")
    f.close()

with open ("jastrow/data/dataset"+str(dataset_ind)+"/JastrowParams", "w") as f:
    f.write(str(len(params["kernel"]))+"\n")
    for param in params["kernel"].flatten():
        f.write(str(param)+"\n")
    f.close()

with open ("jastrow/data/dataset"+str(dataset_ind)+"/JastrowState", "w") as f:
    f.write(str(len(pred_amps))+"\n")
    for amp in pred_amps.flatten():
        f.write(str(amp)+"\n")
    f.close()

with open ("jastrow/data/dataset"+str(dataset_ind)+"/JastrowStateFull", "w") as f:
    f.write(str(len(pred_amps_full))+"\n")
    for amp in pred_amps_full.flatten():
        f.write(str(amp)+"\n")
    f.close()


plt.plot([i for i in range(len(pred_amps))],pred_amps, color = 'r', label = "Jastrow Model")
plt.plot([i for i in range(len(pred_amps))],amps, color = 'b', label = "Exact State")
plt.title("Jastrow Model vs Exact State, Overlap = "+str(np.abs(jnp.vdot(pred_amps, amps))))
plt.legend()    
plt.savefig("jastrow/data/dataset"+str(dataset_ind)+"/JastrowStatePlot")
plt.clf()