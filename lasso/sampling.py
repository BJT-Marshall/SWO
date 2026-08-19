# Computational Basis Wavefunction Sampling

import numpy as np

def basis_measurement_sampling(wf, num_samples, unique = False):
    """
    Samples wavefunction amplitudes according to thier probability of measurement in the computational basis.
    Used to generate more realistic sampling sets to be used as training data for supervised wavefunction learning schemes.

    :param wf: List of wavefunction amplitudes. Must be normalised such that <wf|wf> = 1
    :type wf: list
    :param num_samples: Number of samples to take from the wavefunction list
    :type num_samples: int
    :param unique: Whether to uniquely sample wavefunction amplitudes. This is 'un-physical' however convinient for testing supervised wavefunction learning.
    :type unique: bool
    :returns samples: List of wavefunction indices sampled according to the probability distribution p(x) = <x|wf><wf|x>
    :rtype: list
    :returns s_dict: Dictionary of sampled index counts
    :rtype: dict
    """
    
    dist = []
    s_dict = {}
    wf = list(wf)
    for x in wf:
        dist.append(np.conjugate(x)*x)

    samples = list(np.random.choice(a=np.arange(0,len(wf)), size=int(num_samples), p=dist, replace = not unique))
    for s in range(len(samples)):
        samples[s] = int(samples[s])
    

    for element in samples:
        s_dict[str(element)] = samples.count(element)

    return samples, s_dict



def example():

    import netket as nk
    import GPSKet.operator.hamiltonian.J1J2 as j1j2

    ha = j1j2.get_J1_J2_Hamiltonian(Lx=4, Ly = 2, J2=0, sign_rule=[True,False], on_the_fly_en=False)
    e, state = nk.exact.lanczos_ed(ha, compute_eigenvectors=True, k=1)
    amps = list(state.flatten())

    print("Amplitudes:")
    print(amps)
    samps, s_dict = basis_measurement_sampling(amps,int(len(amps)/2))
    print("35 Sampled Indices:")
    print(samps)
    print("Sampled Index Counts:")
    print(s_dict)

    return None