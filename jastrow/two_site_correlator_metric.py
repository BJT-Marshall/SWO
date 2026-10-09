import itertools
import numpy as np
from GPSKet.operator.hamiltonian import J1J2
import netket as nk
from GPSKet.models import qGPS
from GPSKet.nn.initializers import normal
import jax.numpy as jnp
import re


def extract_cA(f_values, L):
    """
    Extract Walsh-Hadamard coefficients c_A.

    Parameters
    ----------
    f_values : ndarray, shape (2**L,)
        f(s)=log|psi(s)| evaluated on every spin configuration.

        Ordering convention:
        config index k corresponds to the binary representation of k,
        with bit=0 -> spin=-1 and bit=1 -> spin=+1.

    L : int
        Number of spins.

    Returns
    -------
    cA : dict
        Maps subsets A (as tuples of site indices) to coefficients.
    """

    configs = np.array([
        [1 if (n >> i) & 1 else -1 for i in range(L)]
        for n in range(2**L)
    ])

    cA = {}

    for r in range(L + 1):
        for A in itertools.combinations(range(L), r):

            if len(A) == 0:
                feature = np.ones(2**L)
            else:
                feature = np.prod(configs[:, A], axis=1)

            coeff = np.mean(f_values * feature)
            cA[A] = coeff

    return cA

def get_qGPS_params(file_name):
    """Reads in and formats the qGPS parameters from a file. The parameters are expected to be stored in a flattened format, 
    with the first line of the file containing the dimensions of the parameter array (D, M, L). The function returns the 
    parameters as a reshaped JAX array with shape (D, M, L)."""

    qGPS_params = []
    
    with open(file_name, "r") as f:
        shape = f.readline().strip().split()
        dims = [int(dim) for dim in shape] #[D,M,L]
        n = dims[0]*dims[1]*dims[2]  # Total number of parameters
        for i in range(n):
            qGPS_params.append(float(f.readline().strip()))
        qGPS_params = jnp.array(qGPS_params)
        qGPS_params = qGPS_params.reshape(dims[0], dims[1], dims[2])  # Reshape to (D,M,L)
        f.close() 
    
    return qGPS_params, dims

def get_jastrow_params(file_name):
    """Reads in the Jastrow parameters from a file. The first line of the file should contain the number of parameters, 
    and each subsequent line should contain one parameter value. The function returns the parameters as a list of floats."""

    jastrow_params = []
    
    with open(file_name, "r") as f:
        n = int(f.readline().strip())
        for i in range(n):
            jastrow_params.append(complex(f.readline().strip()))
        f.close() 
    
    return jastrow_params


def trailing_number(s):

    match = re.search(r'(\d+)$', s)

    return int(match.group(1)) if match else None

def compute_P2(qGPS_test_folder,log_amps_J_file = "jastrow/data/dataset2/JastrowStateFull", params_J_file = "jastrow/data/dataset2/JastrowParams", print_steps = False):
    """Computes the P2 metric for the states produced from both the Jastrow model and qGPS model. Used to compare how "pair-wise" the state
    is that the qGPS model produces."""

    # Read in qGPS and Jastrow model params
    
    qGPS_params, shape = get_qGPS_params(qGPS_test_folder+"/Epsilon_pred_r")
    jastrow_params = get_jastrow_params(params_J_file)

    # System Setup
    with open(qGPS_test_folder+"/SetupTest"+str(trailing_number(qGPS_test_folder))) as f:
        Lx =  int(f.readline().strip().split()[-1])
        Ly = int(f.readline().strip().split()[-1])
        f.close()

    L = Lx*Ly
    
    if Lx == 1:
        Lx = None
        g = nk.graph.Chain(Ly, pbc = True)
    elif Ly ==1:
        Ly = None
        g = nk.graph.Chain(Lx, pbc = True)
    else:
        g = nk.graph.Grid([Lx,Ly], pbc = True)

    ha = J1J2.get_J1_J2_Hamiltonian(Lx=Lx, Ly = Ly, J2=0, sign_rule=[True,False], on_the_fly_en=False)
    full_configs = [[(n >> i) & 1 for i in range(7, -1, -1)] for n in range(2**L)]


    # qGPS vs setup

    model_R = qGPS(ha.hilbert, shape[1], init_fun=normal(1.0e-3), dtype=float)
    sa_R = nk.sampler.MetropolisExchange(ha.hilbert, graph=g)
    vs_R = nk.vqs.MCState(sa_R, model_R, seed=1, sampler_seed=1) #apply_fun=model_R.apply) 

    # Jastrow vs setup

    model = nk.models.Jastrow()
    vs = nk.vqs.FullSumState(hilbert=ha.hilbert, model=model)
    vs.init_parameters()


    # Generate full lists of log(|psi(x)|) for all spin configs x from both the qGPS and Jastrow model parameters

    absolute_log_amps_J = []
    with open(log_amps_J_file, "r") as f:
        n = int(f.readline().strip())
        for i in range(n):
            absolute_log_amps_J.append(complex(f.readline().strip()).real)


    absolute_log_amps = vs_R._apply_fun({"params": {"epsilon": qGPS_params}},ha.hilbert.local_indices_to_states(jnp.array(full_configs)))
    absolute_log_amps /= jnp.linalg.norm(absolute_log_amps)

    # Decompose both amplitude sets using the Hadamard-Walsh decomp to get correlation coefficients

    c_A_values = extract_cA(absolute_log_amps,L)
    c_A_values_J = extract_cA(absolute_log_amps_J,L)


    # Jastrow coeff investigation

    two_spin_coefs_J = [abs(float(v))**2 for k, v in  c_A_values_J.items() if len(k) == 2 or len(k) == 1]
    other_spin_coefs_J = [abs(float(v))**2 for k, v in c_A_values_J.items() if k!=()]

    if print_steps:
        print("Two site correlator coefficients sum: ")
        print(sum(two_spin_coefs_J))
        print("All correlator coefficients sum: ")
        print(sum(other_spin_coefs_J))

    P_J = sum(two_spin_coefs_J)/sum(other_spin_coefs_J)
    print("Jastrow P2: "+str(P_J))


    #qGPS coeff investigation

    two_spin_coefs = [abs(float(v))**2 for k, v in  c_A_values.items() if len(k) == 2 or len(k) == 1]
    other_spin_coefs = [abs(float(v))**2 for k, v in c_A_values.items() if k!=()]

    if print_steps:
        print("Two site correlator coefficients sum: ")
        print(sum(two_spin_coefs))
        print("All correlator coefficients sum: ")
        print(sum(other_spin_coefs))

    P_qGPS = sum(two_spin_coefs)/sum(other_spin_coefs)
    print("qGPS P2: "+str(P_qGPS))


    # Perform the Hadamard-Walsh decomp ON THE JASTROW state to check that it correctly decomposes into just two body terms and then 
    # this result can be used as a reference to guide how to best compare qGPS learned jastrow states to the jastrow states themselves.

    if print_steps:
        print("Norms:")

        print("Jastrow:")
        check_norm = jnp.array([abs(float(v)) for k, v in c_A_values_J.items()])
        print(jnp.linalg.norm(check_norm))
        other_check_norm = jnp.array([abs(float(v)) for k, v in c_A_values_J.items() if k!=()])
        print(jnp.linalg.norm(other_check_norm))
        print("Empty Set Term:")
        print(c_A_values_J[()])

        print("qGPS:")
        check_norm = jnp.array([abs(float(v)) for k, v in c_A_values.items()])
        print(jnp.linalg.norm(check_norm))
        other_check_norm = jnp.array([abs(float(v)) for k, v in c_A_values.items() if k!=()])
        print(jnp.linalg.norm(other_check_norm))
        print("Empty Set Term:")
        print(c_A_values[()])

    return P_J, P_qGPS



p_j,p_q = compute_P2("jastrow/Tests/Test47")


#with open("P2ForTests","w") as f:
    #for i in range(1,39):
        #p_j,p_q = compute_P2("jastrow/Tests/Test"+str(i))
        #f.write("Test "+str(i)+" P2 Value: "+str(p_q)+"\n")
    #f.close()


