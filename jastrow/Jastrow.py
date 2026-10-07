#Jastrow Model Generation for Small Test Systems

import netket as nk
import jax.numpy as jnp
from GPSKet.operator.hamiltonian import J1J2
import numpy as np
import matplotlib.pyplot as plt
import optax
import jax

Lx = 4
Ly = 2

hilbert = nk.hilbert.Spin(s=0.5, N=Lx*Ly)
ha = J1J2.get_J1_J2_Hamiltonian(Lx=Lx, Ly = Ly, J2=0, sign_rule=[True,False], on_the_fly_en=False)
g = nk.graph.Grid([Lx,Ly], pbc = True)
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

# Enforce global phase convention by making the largest amplitude positive
max_idx = np.argmax(np.abs(pred_amps))
if pred_amps[max_idx] < 0:
    pred_amps *= -1

#Save the jastrow model parameters to a file

with open ("jastrow/data/JastrowParams", "w") as f:
    f.write(str(len(params["kernel"]))+"\n")
    for param in params["kernel"].flatten():
        f.write(str(param)+"\n")
    f.close()

with open ("jastrow/data/JastrowState", "w") as f:
    f.write(str(len(pred_amps))+"\n")
    for amp in pred_amps.flatten():
        f.write(str(amp)+"\n")
    f.close()


plt.plot([i for i in range(len(pred_amps))],pred_amps, color = 'r', label = "Jastrow Model")
plt.plot([i for i in range(len(pred_amps))],amps, color = 'b', label = "Exact State")
plt.title("Jastrow Model vs Exact State, Overlap = "+str(np.abs(jnp.vdot(pred_amps, amps))))
plt.legend()    
plt.savefig("jastrow/data/JastrowStatePlot")
plt.clf()