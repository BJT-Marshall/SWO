import numpy as np
import netket as nk
import scipy as spy

import GPSKet

from GPSKet.models import qGPS

import jax.numpy as jnp

from GPSKet.supervised.supervised_qgps import QGPSLogSpaceFit
from GPSKet.nn.initializers import normal

import jax


import optax

import GPSKet.operator.hamiltonian.J1J2 as j1j2

import sklearn
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

import matplotlib.pyplot as plt

from functools import partial
from random import randint
from random import sample
from quspin.basis import spin_basis_general  # spin basis constructor
from quspin.operators import hamiltonian # Hamiltonians and operators
#from Chunk_Calling_qGPS import chunk_qGPS_call, apply_qGPS, final_wavefunction


#TODO Figure out why original transformation from fitted parameters to wavefunction amplitudes (using the vs_R and vs_I objects) doesnt work


def system_setup(hilbert, graph, M, seed = None, smp_seed = None):
    """
    Produces variational quantum state objects (:python:`netket.vqs.MCState`) initialised to fit states of an inputted hilbert space
    using the qGPS variational model with support dimension :python:`M`.

    :param hilbert: Hilbert space of system of interest implemented as a :python:`netket.hilbert` object or subclass.
    :type hilbert: netket.hilbert
    :param graph: Graph of the system of interest.
    :type graph: netket.graph
    :param M: Support dimension of the qGPS model, defines the number of the model parameters as :math:`n = M\\times D\\times L`.
    :type M: int or list

    Optional Parameters:

    :param seed: Random seed used to generate initial parameters for the qGPS model, defaults to a random seed if none are provided.
    :type seed: int
    :param smp_seed: Random seed used to generate samples from the hilbert space, defaults to a random seed if none are provided.
    :type smp_seed: int

    :returns vs_R: Variational quantum state used to fit the real components of wavefunction training data
    :returns vs_I: Variational quantum state used to fit the imaginary components of wavefunction training data
    :rtype netket.vqs.MCState:
    """

    if isinstance(M,int):
        M = [M,M]    
    model_R = qGPS(hilbert, M[0], init_fun=normal(1.0e-3), dtype=float)
    model_I = qGPS(hilbert, M[1], init_fun=normal(1.0e-3), dtype=float)

    sa_R = nk.sampler.MetropolisExchange(hilbert, graph=graph, n_chains_per_rank = 50)
    sa_I = nk.sampler.MetropolisExchange(hilbert, graph=graph, n_chains_per_rank = 50)

    vs_R = nk.vqs.MCState(sa_R, model_R, seed=seed, sampler_seed=smp_seed)
    vs_I = nk.vqs.MCState(sa_I, model_I, seed=seed, sampler_seed=smp_seed)

    return vs_R, vs_I


def rescale_weights(weights):
    """
    Rescales the weights computed through an iteration of the :python:`sklearn.linear_model.Lasso` fitting function 
    such that pairs of weights corresponding to the pair of spin configurations of a single site and support configuration
    made redundant (set to 0) are replaced with weights of value 1. All other pairs of weights are rescaled to have a 
    maximum magnitude equal to 1.

    Example:

    :math:`w_{0,i} = w_{1,i} = 0 \\rightarrow w'_{0,i} = w'_{1,i} = 1`
    :math:`w_{0,i} = -2.0, w_{1,i} = 1.6 \\rightarrow w'_{0,i} = -1, w'_{1,i} = 0.8`

    
    :param weights: Fitted weights as computed by the :python:`sklearn.linear_model.Lasso` fitting function.
    :type weights: numpy.ndarray
    :returns weights: Rescaled weights according to the scheme described above.
    :rtype weights: numpy.ndarray
    :returns rescalings: Rescaling factors calculated according to the scheme described above.
    :rtype rescalings: list
    """

    m_ = int(len(weights)/2)
    odd_indices = [2*n for n in range(m_)]
    rescalings = [max([abs(weights[i]), abs(weights[i+1])]) for i in odd_indices]
    for index in odd_indices:
        if rescalings[int(index/2)] == 0:
            weights[index] = 1
            weights[index+1] = 1
        else:
            weights[index] = weights[index]/rescalings[int(index/2)]
            weights[index+1] = weights[index+1]/rescalings[int(index/2)]

    return weights, rescalings


def rescale_parameters(learning_obj, rescalings, site):
    """
    Rescales the fitted weights inside of a seperate parameter set, :python:`learning_pred`, used for final  model predictions and 
    tracking predicted values for training configurations throughout subsequent iterations for use in scaling training data.

    Effectively 'undoes' the rescaling performed by :python:`rescale_weights` on a copy of the fitted parameters stored in a 
    seperate parameter array.

    
    :param learning_obj: :python:`QGPSLogSpaceFit` object containing the fitted parameters.
    :type learning_obj: QGPSLogSpaceFit
    :param rescalings: Rescaled fitted weights as returned by the :python:`rescale_weights` function.
    :type rescalings: list
    :param site: The site being currently fit, corresponds to the final index of interest for rescaling in the parameter array.
    :type site: int
    :returns learning_pred: :python:`QGPSLogSpaceFit` object containing the rescaled fitted parameters.
    :rtype QGPSLogSpaceFit:
    """

    rescaled_epsilon = jnp.array([element for element in jnp.array(learning_obj.epsilon).flatten()]).reshape(
        learning_obj.epsilon.shape[0],
        learning_obj.epsilon.shape[1],
        learning_obj.epsilon.shape[2])
    
    for m in range(rescaled_epsilon.shape[1]):
        for d in range(rescaled_epsilon.shape[0]):
            #For the current reference site multiply back in the scalings
            rescaled_epsilon = rescaled_epsilon.at[d,m,site].multiply(rescalings[m])

    #Seperate learning object just to hold the rescaled epsilon tensor for predicting log amplitudes
    learning_pred = QGPSLogSpaceFit(
        jnp.array(rescaled_epsilon)
        )
    
    return learning_pred


def lasso_wf_optimization(vs_R, vs_I, training_data, training_data_configs, iterations, regularization_penalty, scaling):
    """
    #TODO Finish documentation

    :param vs_R: Variational quantum state object as returned by the :python:`system_setup` method for fitting real wavefunction data.
    :type vs_R: netket.vqs.MCState
    :param vs_I: Variational quantum state object as returned by the :python:`system_setup` method for fitting imaginary wavefunction data.
    :type vs_I: netket.vqs.MCState
    :param training_data: Set of wavefunction training data containing LOG amplitudes.
    :type training_data: list
    :param training_data_configs: Set of hilbert space configurations mapping the training data to the correct states.
    :type training_data_configs: list
    :param iterations: Number of iterations of LASSO sweeping to perform when fitting model
    :type iterations: int or list
    :param regularization_penalty: Regularization penalty parameter alpha in the LASSO loss function.
    :type regularization_parameter: int or list
    :param scaling: Boolean flag to determine whether data is scaled to prioritise fitting dominant amplitudes in the training data.
    :type scaling: bool
    """
    
    epsilon_R = np.array(vs_R.parameters["epsilon"])  # reset the epsilon tensor
    learning_R = QGPSLogSpaceFit(
        epsilon_R
    )  # The way of interfacing the learning model with the qGPS state

    epsilon_I = np.array(vs_I.parameters["epsilon"])  # reset the epsilon tensor
    learning_I = QGPSLogSpaceFit(
        epsilon_I
    ) # The way of interfacing the learning model with the qGPS state
    
    #Dividing training data into real and imaginary sets
    log_amps_R = jnp.real(training_data)
    log_amps_I = jnp.imag(training_data) 

    #LASSO learning models
    if isinstance(regularization_penalty,float) or isinstance(regularization_penalty,int):
        regularization_penalty = [regularization_penalty,regularization_penalty]

    lasso_model_R = Lasso(alpha = regularization_penalty[0], fit_intercept=False, warm_start=True)
    lasso_model_I = Lasso(alpha = regularization_penalty[1], fit_intercept=False, warm_start=True)
    
    if isinstance(iterations,int):
        iterations = [iterations, iterations]

    #----------------------------------------------REAL FITTING LOOP----------------------------------------------------

    #Data pre-processing
    fit_data_R_pre = log_amps_R
    mean_data_R = ((jnp.exp(log_amps_R)/jnp.linalg.norm(jnp.exp(log_amps_R)))**2).flatten()*fit_data_R_pre
    fit_data_R_pre -=jnp.sum(mean_data_R)
    fit_data_R = fit_data_R_pre

    #Fitting loop for real training data components
    for i in range(iterations[0]):
        if scaling:
            if i != 0:
                log_estimate = vs_R._apply_fun({"params": {"epsilon": prediction_model_R.epsilon}}, training_data_configs)
            else:
                log_estimate = vs_R._apply_fun({"params": {"epsilon": learning_R.epsilon}}, training_data_configs)
            log_scalings = log_estimate - jnp.log(jnp.linalg.norm(jnp.exp(log_estimate)))
            scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)
            fit_data_R = scalings.flatten()*fit_data_R_pre

        for site in np.arange(epsilon_R.shape[-1]):

            learning_R.reset()
            learning_R.ref_sites = site

            #scaling: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling == True:
                K_R=learning_R.set_kernel_mat(update_K=True, confs=training_data_configs) #sampled amplitudes converted to configs_list as demanded by the 'set_kernel_mat' method
                feature_vector_R = np.array(scalings)*K_R
            else:
                K_R=learning_R.set_kernel_mat(update_K=True, confs=training_data_configs) #sampled amplitudes converted to configs_list as demanded by the 'set_kernel_mat' method
                feature_vector_R = K_R

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            fit_R = lasso_model_R.fit(X=feature_vector_R, y=fit_data_R).coef_
            optimal_weights_R, rescalings_R = rescale_weights(fit_R)
            learning_R.weights = optimal_weights_R
           
            #Update the weights and the epsilon tensor held in the learning object.
            learning_R.valid_kern = abs(np.diag(K_R.conj().T.dot(K_R))) > learning_R.kern_cutoff
            learning_R.update_epsilon_with_weights()

            prediction_model_R = rescale_parameters(learning_R, rescalings_R, site)

    #----------------------------------------------IMAGINARY FITTING LOOP------------------------------------------------
            
    #Data pre-processing
    fit_data_I_pre = log_amps_I
    phase_shift = jnp.sum(((jnp.exp(log_amps_I)/jnp.linalg.norm(jnp.exp(log_amps_I)))**2).flatten()*fit_data_I_pre)
    fit_data_I_pre -=phase_shift

    #if scaling, generate scalings from the fitted real training data components
    if scaling:
        estimated_log_amps_R = vs_R._apply_fun({"params": {"epsilon": prediction_model_R.epsilon}}, training_data_configs)
        log_scalings = estimated_log_amps_R - jnp.log(jnp.linalg.norm(jnp.exp(estimated_log_amps_R)))
        scalings = jnp.expand_dims(jnp.exp(log_scalings), -1)
        fit_data_I = scalings.flatten()*fit_data_I_pre

    #Fitting loop for imaginary training data components
    for i in range(iterations[1]):
        for site in np.arange(epsilon_R.shape[-1]):

            learning_I.reset()
            learning_I.ref_sites = site

            #scaling: target data and feature vector both individually scaled by |psi|_predicted at each iteration
            if scaling:
                K_I=learning_I.set_kernel_mat(update_K=True, confs=training_data_configs) #sampled amplitudes converted to configs_list as demanded by the 'set_kernel_mat' method 
                feature_vector_I = jnp.array(scalings)*K_I 
            else:
                K_I=learning_I.set_kernel_mat(update_K=True, confs=training_data_configs) #sampled amplitudes converted to configs_list as demanded by the 'set_kernel_mat' method
                feature_vector_I = K_I
                fit_data_I = fit_data_I_pre

            #Fitting the model (Computes the optimal weights 'w' that fits the feature vector to the fit data)
            fit_I = lasso_model_I.fit(X=feature_vector_I, y=fit_data_I).coef_
            optimal_weights_I, rescalings_I = rescale_weights(fit_I)
            learning_I.weights = optimal_weights_I
    
            #Update the weights and the epsilon tensor held in the learning object.
            learning_I.valid_kern = abs(np.diag(K_I.conj().T.dot(K_I))) > learning_I.kern_cutoff
            learning_I.update_epsilon_with_weights()

            prediction_model_I = rescale_parameters(learning_I, rescalings_I, site)

    return prediction_model_R, prediction_model_I, phase_shift


def apply_model(vs_R, vs_I, params_R, params_I, phase_shift, configs):
    """
    Generates the set of predicted, normalised, wavefunction amplitudes corresponding to an inputted set of 
    local hilbert space configurations.

    :param vs_R: Variational quantum state as returned by :python:`lasso_wf_optimization` containing fitted parameters for the real
    component of the training data.
    :type vs_R: netket.vqs.MCState
    :param vs_I: Variational quantum state as returned by :python:`lasso_wf_optimization` containing fitted parameters for the imaginary
    component of the training data.
    :type vs_I: netket.vqs.MCState
    :param params_R: The fitted parameters for the real component of the training data.
    :type params_R: numpy.ndarray
    :param params_I: The fitted parameters for the imaginary component of the training data.
    :type params_I: numpy.ndarray
    :param phase_shift: The phase shift used to adjust the imaginary component of the training data as returned by :python:`lasso_wf_optimization`.
    :type phase_shift: float
    :param configs: A list of local hilbert space configurations corresponding to the wavefunction amplitudes to be generated.
    :type configs: list
    :returns predicted_amps: The generated, normalised, wavefunction amplitudes corresponding to the inputted local hilbert space configurations.
    :rtype numpy.array:
    """

    #TODO Figure out why this implementation generates constant wavefunctions regardless of fitted parameters
    #prediction_R = vs_R._apply_fun({"params": {"epsilon": params_R}}, configs)
    #prediction_I = vs_I._apply_fun({"params": {"epsilon": params_I}}, configs)

    prediction_R = jnp.array(apply_qGPS_manual(params_R, configs))
    prediction_I = jnp.array(apply_qGPS_manual(params_I, configs))
    prediction_I += phase_shift

    predicted_log_amps = jnp.array([prediction_R[i]+1j*prediction_I[i] for i in range(len(prediction_R))])
    predicted_amps = jnp.exp(predicted_log_amps)/jnp.linalg.norm(jnp.exp(predicted_log_amps))

    return predicted_amps


def apply_qGPS_manual(epsilon, configs):
    """
    Generates predicted log wavefunction amplitudes for an inputted set of local hilbert space configurations from a qGPS model 
    according to the equation: :math:`\\log{(\\psi(\\vec{x}))} = \\sum_{x'=1}^M \\prod_{i=1}^L \\epsilon_{\\vec{x}_i,x',i}`.

    :param epsilon: qGPS parameter set used to predict log wavefunction amplitudes.
    :type epsilon: numpy.ndarray
    :param configs: A list of local hilbert space configurations to generate log wavefunction amplitudes for.
    :type configs: list
    :returns log_amps: A list of log wavefunction amplitudes corresponding to the local hilbert space configurations generated by the inputted parameter set.
    :rtype list:
    """
    
    #epsilon.shape = (D,M,L)
    M = epsilon.shape[1]
    L = epsilon.shape[2]

    log_amps = []
    
    for conf in configs:
        conf_log_amp = 0
        for m in range(M):
            product = 1
            for l in range(L):
                product *= epsilon[int(conf[l])][m][l]
            conf_log_amp += product
        log_amps.append(conf_log_amp) 

    return log_amps