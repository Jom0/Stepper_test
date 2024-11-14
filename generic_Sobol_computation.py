import os
import numpy as np
import itertools
import time
import copy
from scipy.stats import multivariate_normal
import gc
import tracemalloc
import multiprocessing as mp
import functools as ft
import openturns as ot
import openturns.experimental as otexp
import sys


A1 = np.random.rand(10, 10)
cov1 = np.dot(A1, A1.transpose())
A2 = np.random.rand(10, 10)
cov2 = np.dot(A2, A2.transpose())
A3 = np.random.rand(10, 10)
cov3 = np.dot(A3, A3.transpose())

mu1 = np.random.rand(10)*5
mu2 = np.random.rand(10)*5
mu3 = np.random.rand(10)*5

normal1 = multivariate_normal(mu1, cov1)
normal2 = multivariate_normal(mu2, cov2)
normal3 = multivariate_normal(mu3, cov3)

#test_function = lambda x: 10000*(-normal1.pdf(x)-normal2.pdf(x)-normal3.pdf(x))

def Ackley(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    d = 10
    square_sum = np.sum([x_i**2 for x_i in x])
    cos_sum = np.sum([np.cos(c*x_i) for x_i in x])
    return -a*np.exp(-b*np.sqrt(1/d*square_sum))-np.exp(1/d*cos_sum)+a+np.exp(1)

mini1 = [1]*10
mini2 = [3,2,1,3,2,1,3,2,1,3]

mini3 = [1,3,2,1,3,2,1,3,2,1]
#test_function = lambda x: Ackley(100*(x-mini1))+Ackley(100*(x-mini2))+Ackley(100*(x-mini3))

def Ackley2(x,y):
    a = 20
    b = 0.2
    c = 2*np.pi
    return -a*np.exp(-b*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(c*x)+np.cos(c*y)))+a+np.exp(1)

def normal_pdf(x,y,mu1,mu2,sig1,sig2):
    return 1/(2*np.pi*sig1*sig2)*np.exp(-0.5*(((x-mu1)/sig1)**2)+((y-mu2)/sig2)**2)

#test_function = lambda x,y: Ackley2(x-10,y-10)+Ackley2(x-50,y-90)
#test_function = lambda x,y: Ackley2(x-10,y-300)
#test_function = lambda x,y: -normal_pdf(x,y,5,90,30,20)
ALPHA = 1.1
test_function = lambda x,y: -ALPHA*x-y



f = lambda x : test_function(x[0],x[1])

#QoI_threshold = 23

#QoI_threshold = -1e4

QoI_threshold = -100

def true_failure_proba(lambda_1,lambda_2):
    """
    True failure proba for -2x-y analytical function
    """
    # if lambda_1/2!=lambda_2:
    #     return lambda_2/(lambda_2-lambda_1/2)*(np.exp(-lambda_1*(-QoI_threshold)/2)-(lambda_1/(2*lambda_2))*np.exp(-lambda_2*(-QoI_threshold)))
    # print("lambda_1/2=lambda_2")
    # return np.exp(-lambda_1*(-QoI_threshold)/2)*(-QoI_threshold+2/lambda_1)

    if lambda_1/ALPHA!=lambda_2:
        return lambda_2/(lambda_2-lambda_1/ALPHA)*(np.exp(-lambda_1*(-QoI_threshold)/ALPHA)-(lambda_1/(ALPHA*lambda_2))*np.exp(-lambda_2*(-QoI_threshold)))
    return np.exp(-lambda_1*(-QoI_threshold)/2)*(-QoI_threshold+2/lambda_1)

def get_X_param():
    # to do with XML file parsing
    """
    Get input exponential parameters
    
    """
    #exp_param = [0.1,0.1,0.1,0.01,0.01,0.01,0.001,0.001,0.001,0.5]
    exp_param = [0.1,0.1]
    return [[i,f"parameter {i}",exp_param[i]] for i in range(len(exp_param))]
        
def draw_X(parameter_list):
    """
    Draw input value X associated to parameter_list
    This corresponds to one simulation
    """
    # list of couples of indices of transition associated with a delay
    parameters = []
    for param in parameter_list:
        parameters.append([param[0],np.random.default_rng().exponential(scale=1/param[2])])
    return parameters



def M_psi(inputs) :
    # could be modified to take into account multiple times and observers
    """
    Run the execution of the AltaRica simulation corresponding to delays until max_time
    and return the final obs_name value, and the precise time of failure of the system

    The observer takes boolean values so we translate "true" to 1 and "false" to 0
    """
    output = f(np.array(inputs)[:,1])
    QoI = int(output<=QoI_threshold)
    return QoI, output

def M_psi_simpler(inputs) :
    # could be modified to take into account multiple times and observers
    """
    Run the execution of the AltaRica simulation corresponding to delays until max_time
    and return the final obs_name value, and the precise time of failure of the system

    The observer takes boolean values so we translate "true" to 1 and "false" to 0
    """
    output = f(inputs)
    QoI = int(output<=QoI_threshold)
    return QoI, output

def Monte_Carlo(parameter_list,n_run):
    """
    Run a Monte-Carlo simulation of size n_run
    Returns the drawn X with the associated black box result
    """
    # drawn X
    X = np.zeros((n_run,len(parameter_list)))
    # results of simulation
    Y = np.zeros(n_run)
    # fail times
    outputs = np.zeros(n_run)
    for simu_index in range(n_run):
        inputs = draw_X(parameter_list)
        X[simu_index] = [couple[1] for couple in inputs]
        simu = M_psi(inputs)
        Y[simu_index] = simu[0]
        outputs[simu_index] = simu[1]
    return X,Y,outputs



def draw_X_IS_g(draw_g,parameter_list):
    """
    Draw delays using a list of function that generates samples draw_g
    """
    # sample delays according to modified g
    delays = []
    for tr in parameter_list:
        sample = draw_g[tr[0]]()
        delays.append([tr[0],sample])
    return delays

# def sample_IS(draw_g,parameter_list,n_run):
#     """
#     Run an simulation of size n_run
#     Sampling done along draw_g
#     Returns the drawn delays with the associated AltaRica simulation result
#     """
#     # drawn delays
#     X = np.zeros((n_run,len(parameter_list)))
#     # results of simulation
#     Y = np.zeros(n_run)
#     # fail times
#     fail_times = np.zeros(n_run)
#     # for simu_index in range(n_run):
#     #     delays = draw_X_IS_g(draw_g,parameter_list)
#     #     #print(f"X[simu_index] : {X[simu_index]}")
#     #     #print(f"[couple[1] for couple in delays] : {[couple[1] for couple in delays]}")
#     #     X[simu_index] = np.array([couple[1] for couple in delays])
#     #     simu = M_psi(delays)
#     #     Y[simu_index] = simu[0]
#     #     fail_times[simu_index] = simu[1]

#     # Vectorized delay generation
#     delays_list = [draw_X_IS_g(draw_g, parameter_list) for _ in range(n_run)]

#     # Extract the second element from each "couple" in the delays for all simulations
#     X = np.array([[couple[1] for couple in delays] for delays in delays_list])

#     # Run M_psi in a vectorized manner for all simulations
#     simu_results = np.array([M_psi(delays) for delays in delays_list])

#     # Assign results
#     Y = simu_results[:, 0]  # First element of each tuple
#     fail_times = simu_results[:, 1]  # Second element of each tuple
#     del delays_list
#     gc.collect()
#     return X,Y,fail_times    


def draw_X_IS_g_vectorized_old(draw_g, parameter_list, n_run):
    """
    Draw delays using a list of function that generates samples draw_g
    Vectorized version for better performance.
    """
    n_params = len(parameter_list)
    
    # Preallocate memory for delays
    delays = np.zeros((n_run, n_params))

    # Sample all delays in a vectorized manner
    for i, tr in enumerate(parameter_list):
        delays[:, i] = [draw_g[tr[0]]() for _ in range(n_run)]
    
    return delays

def sample_IS_old(draw_g, parameter_list, n_run):
    """
    Run a simulation of size n_run.
    Sampling done along draw_g in a vectorized manner for efficiency.
    Returns the drawn delays with the associated AltaRica simulation result.
    """
    # Draw delays for all simulations in a vectorized manner
    X = draw_X_IS_g_vectorized_old(draw_g, parameter_list, n_run)
    
    # Preallocate memory for results
    Y = np.zeros(n_run)
    fail_times = np.zeros(n_run)

    # Run M_psi in a vectorized manner for all simulations
    simu_results = np.array([M_psi_simpler(X[i, :]) for i in range(n_run)])

    # Assign results
    Y[:] = simu_results[:, 0]  # First element of each tuple (vectorized assignment)
    fail_times[:] = simu_results[:, 1]  # Second element of each tuple

    return X, Y, fail_times



def compute_pf_cv(g,parameter_list,X,fail_times,obs_time):
    """
    Compute failure probability and coefficient of variation for an IS simulations until obs_time
    """
    n_run = len(X)
    # table for the computation of the multiplication factors
    f_g_array = np.zeros(n_run)
    for simu_index in range(n_run):
        # if failure before obs_time, compute f/g
        if fail_times[simu_index]<obs_time:
            f_g = 1
            for tr_index in range(len(parameter_list)):
                lbd = parameter_list[tr_index][2]
                x = X[simu_index][tr_index]
                f_g*= (lbd*np.exp(-lbd*x)/g[parameter_list[tr_index][0]](x))
            f_g_array[simu_index] = f_g
    p_f=np.mean(f_g_array)

    # CV estimation
    cv = np.sqrt(np.var(f_g_array))/(np.sqrt(n_run)*p_f)
    return p_f,cv

def compute_pf_cv_alt(lbd_new,g,X,fail_times,obs_time):
    """
    Compute failure probability and coefficient of variation for an IS simulations until obs_time
    with a different lambda that the one used for sampling
    """
    n_run = len(X)
    # table for the computation of the multiplication factors
    f_g_array = np.zeros(n_run)
    for simu_index in range(n_run):
        # if failure before obs_time, compute f/g
        if fail_times[simu_index]<obs_time:
            f_g = 1
            for tr_index in range(len(lbd_new)):
                lbd = lbd_new[tr_index]
                x = X[simu_index][tr_index]
                f_g*= (lbd*np.exp(-lbd*x)/g[tr_index](x))
            f_g_array[simu_index] = f_g
    p_f=np.mean(f_g_array)

    # CV estimation
    cv = np.sqrt(np.var(f_g_array))/(np.sqrt(n_run)*p_f)
    return p_f,cv


def AIS_CE_old(lbd_start,lbd_0,initial_parameter_list,T,max_time,N,alpha,transition_index_list):
    """
    AIS_CE for p
    """
    # weight parameter for weighted std update
    # w = 0.5
    print(initial_parameter_list)
    n_tr = len(initial_parameter_list)
    local_parameter_list = copy.deepcopy(initial_parameter_list)
    for param_index in range(n_tr):
        local_parameter_list[param_index][2] = lbd_0[param_index]
    print(local_parameter_list)
    quant_index = int(np.ceil(alpha*N))
    # initial sampling pdf g
    lbd = lbd_start 
    # temporary lbd
    lbd_temp = lbd.copy()
    # current threshold time
    t_m = max_time
    p_T = 0
    n_resample = 0
    W_array = np.zeros(N)
    X_IS = np.zeros((N,2))
    fail_times_IS = np.zeros(N)
    while t_m>T or p_T>1 or p_T==0.:
        print(f"t_m = {t_m}")
        # sampling density
        if n_resample == 0:
            # Monte Carlo for first smaple
            X_IS,_,fail_times_IS = Monte_Carlo(local_parameter_list,N)
        else:
            draw_f = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd]
            X_IS,_,fail_times_IS = sample_IS_old(draw_f,local_parameter_list,N)
        #print(fail_times_IS)
        # quantile of level alpha
        #print(fail_times_IS)
        #print(quant_index)
        #print(f"proportion of new samples in failure zone: {np.mean(fail_times_IS<T)}")
        sorted_fail_times_IS = sorted(fail_times_IS)
        t_alpha = sorted_fail_times_IS[quant_index]
        #print(sorted_fail_times_IS)
        t_m = max(t_alpha,T)
        # ratio between target density f_x|lbd_0 and g_lbd
        W = lambda x,lbd_w : int(np.all([x[tr_index]>0 for tr_index in transition_index_list]))*np.sqrt(2*np.pi)**n_tr*np.prod([(lbd_0[tr_index]*lbd_w[tr_index,1]) for tr_index in transition_index_list])*np.exp(np.sum([-x[tr_index]*lbd_0[tr_index]+0.5*((x[tr_index]-lbd_w[tr_index,0])/lbd_w[tr_index,1])**2 for tr_index in transition_index_list]))
        # ratio between f_x|lbd_0 and f_x|lbd
        #W = lambda x,lbd_0_w,lbd_w : np.prod([(lbd_0_w[tr_index]/lbd_w[tr_index]) for tr_index in transition_index_list])*np.exp(np.sum([-x[tr_index]*((lbd_0_w[tr_index]-lbd_w[tr_index])) for tr_index in transition_index_list]))
        #W = lambda x,lbd_0,lbd : np.prod([(lbd[tr_index]/lbd_0[tr_index]) for tr_index in range(n_tr)])*np.exp(np.sum([-x[tr_index]*((lbd[tr_index]-lbd_0[tr_index])/(lbd[tr_index]*lbd_0[tr_index])) for tr_index in range(n_tr)]))
        for tr_index in transition_index_list:
            # if Monte Carlo, natural estimator for lbd (empirical mean and variance)
            if n_resample == 0:
                fail_samples_list = []
                for sample_index in range(N):
                    if fail_times_IS[sample_index]<=t_m :
                        fail_samples_list+=[X_IS[sample_index][tr_index]]
                lbd_temp[tr_index,0] = np.mean(fail_samples_list)
                lbd_temp[tr_index,1] = np.std(fail_samples_list)
                del fail_samples_list
            else:
                lbd_mu_num = 0
                lbd_sigma_num = 0
                lbd_denom = 0
                W_i_list = []
                # compute mu
                for sample_index in range(N):
                    if fail_times_IS[sample_index]<=t_m :
                        #print(f"fail time: {fail_times_IS[sample_index]}")
                        #print(f"associated X: {X_IS[sample_index]}")
                        #print(f"fail_time (output value): {fail_times_IS[sample_index]}")
                        #print(f"associated X: {X_IS[sample_index]}")
                        W_i = W(X_IS[sample_index],lbd)
                        W_i_list+=[W_i]
                        lbd_mu_num+=W_i*X_IS[sample_index][tr_index]
                        lbd_denom+=W_i
                lbd_temp[tr_index,0] = lbd_mu_num/lbd_denom
                # compute sigma
                W_i_list_index = 0
                for sample_index in range(N):
                    if fail_times_IS[sample_index]<=t_m :
                        W_i = W_i_list[W_i_list_index]
                        W_i_list_index+=1
                        #if W_i!=0.0:
                        #    print(f"W_i: {W_i}")
                        #    print(f"X: {X_IS[sample_index][tr_index]}")
                        #    print(f"mu: {lbd_temp[tr_index,0]}")
                        lbd_sigma_num+=(W_i*(X_IS[sample_index][tr_index]-lbd_temp[tr_index,0])**2)
                lbd_temp[tr_index,1] = np.sqrt(lbd_sigma_num/lbd_denom)
                del W_i_list
            gc.collect()
            # version with weighted std
            #lbd_temp[tr_index,1] = (1-w)*lbd[tr_index,1]+w*np.sqrt(lbd_sigma_num/lbd_denom)
            #print(f"sigma num : {lbd_sigma_num}")
            #print(f"denom : {lbd_denom}")
        lbd[:] = lbd_temp[:]
        # estimate probability and cv
        W_array = np.zeros(N)
        for sample_index in range(N):
            if fail_times_IS[sample_index]<=T :
                W_array[sample_index]=W(X_IS[sample_index],lbd)
        p_T = np.mean(W_array)
        cv = np.sqrt(np.var(W_array))/(np.sqrt(N)*p_T)
        print(f"current sampling parameters: {lbd}")
        print(f"current probability estimation: {p_T}")
        print(f"current cv estimation: {cv}")
        if t_m<=T:
            print(f"threshold reached, t_m = {t_m}")
        else:
            del W, W_array, X_IS, fail_times_IS
            gc.collect()    
        n_resample+=1
        
    # estimate probability and cv
    print("loop ended")
    print(" ")
    W_array = np.zeros(N)


    #x1_list = []
    #x2_list = []
    for sample_index in range(N):
        if fail_times_IS[sample_index]<=T :
            W_array[sample_index]=W(X_IS[sample_index],lbd)
            #if np.all([X_IS[sample_index][tr_index]>0 for tr_index in transition_index_list]):
                #print(f"sample in failure zone: {X_IS[sample_index]}")
                #x1_list+=[X_IS[sample_index][0]]
                #x2_list+=[X_IS[sample_index][1]]
    #print(f"std of final sample along x1: {np.std(x1_list)}")
    #+print(f"std of final sample along x2: {np.std(x2_list)}")

    p_T = np.mean(W_array)
    if p_T>1:
        raise Exception("Final estimated probability is greater than 1")
    #cv = np.sqrt(np.var(W_array))/(np.sqrt(N)*p_T)
    #gc.collect()
    return p_T,cv,lbd,n_resample

def draw_X_IS_g_vectorized(draw_g, parameter_list, n_run):
    """
    Draw delays using a list of function that generates samples draw_g
    Vectorized version for better performance.
    """
    n_params = len(parameter_list)
    
    # Preallocate memory for delays
    delays = np.zeros((n_run, n_params))

    # Sample all delays in a vectorized manner
    # for i, tr in enumerate(parameter_list):
    #     delays[:, i] = [draw_g[tr[0]]() for _ in range(n_run)]
    
    for i in range(n_run):
        delays[i,:] = draw_g()
    
    return delays

def sample_IS(draw_g, parameter_list, n_run):
    """
    Run a simulation of size n_run.
    Sampling done along draw_g in a vectorized manner for efficiency.
    Returns the drawn delays with the associated AltaRica simulation result.
    """
    # Draw delays for all simulations in a vectorized manner
    X = draw_X_IS_g_vectorized(draw_g, parameter_list, n_run)
    
    # Preallocate memory for results
    Y = np.zeros(n_run)
    fail_times = np.zeros(n_run)

    # Run M_psi in a vectorized manner for all simulations
    simu_results = np.array([M_psi_simpler(X[i, :]) for i in range(n_run)])

    # Assign results
    Y[:] = simu_results[:, 0]  # First element of each tuple (vectorized assignment)
    fail_times[:] = simu_results[:, 1]  # Second element of each tuple

    return X, Y, fail_times

def AIS_CE(lbd_start,lbd_0,initial_parameter_list,T,max_time,N,alpha,transition_index_list):
    """
    AIS_CE for p
    """
    smoothing_parameter = 0.4
    # weight parameter for weighted std update
    # w = 0.5
    print(initial_parameter_list)
    n_tr = len(initial_parameter_list)
    local_parameter_list = copy.deepcopy(initial_parameter_list)
    for param_index in range(n_tr):
        local_parameter_list[param_index][2] = lbd_0[param_index]
    print(local_parameter_list)
    quant_index = int(np.ceil(alpha*N))
    # initial sampling pdf g
    lbd = lbd_start 
    # temporary lbd
    lbd_temp = lbd.copy()
    # current threshold time
    t_m = max_time
    p_T = 0
    n_resample = 0
    W_array = np.zeros(N)
    X_IS = np.zeros((N,2))
    fail_times_IS = np.zeros(N)
    while t_m>T or p_T>1 or p_T==0.:
        print(f"t_m = {t_m}")
        # sampling density
        if n_resample == 0:
            # Monte Carlo for first smaple
            X_IS,_,fail_times_IS = Monte_Carlo(local_parameter_list,N)
            print(f"proportion of new samples in intermediary failure zone: {np.mean(fail_times_IS<t_m)}")
            # if no samples in failure zone, rerun monte carlo until samples in failure zone
            if np.all(fail_times_IS>=t_m):
                while np.all(fail_times_IS>=t_m):
                    print("no samples in intermediary zone, Monte Carlo resampling")
                    X_IS,_,fail_times_IS = Monte_Carlo(local_parameter_list,N)
                print("sampling with samples in intermediary failure zone found")
                print(f"proportion of new samples in intermediary failure zone: {np.mean(fail_times_IS<t_m)}")
        else:
            print(f"lbd[0] : {lbd[0]}")
            draw_f = (lambda lbd_t = lbd : np.random.default_rng().multivariate_normal(mean = lbd_t[0], cov = lbd_t[1]))
            X_IS,_,fail_times_IS = sample_IS(draw_f,local_parameter_list,N)
            print(f"proportion of new samples in intermediary failure zone: {np.mean(fail_times_IS<t_m)}")
            print(f"proportion of new samples in failure zone: {np.mean(fail_times_IS<T)}")
        sorted_fail_times_IS = sorted(fail_times_IS)
        t_alpha = sorted_fail_times_IS[quant_index]
        #print(f"sorted fail_times : {sorted_fail_times_IS}")
        t_m = max(t_alpha,T)
        # ratio between target density f_x|lbd_0 and g_lbd
        
        W = lambda x,lbd_w : int(np.all([x[tr_index]>0 for tr_index in transition_index_list]))*np.sqrt(2*np.pi)**n_tr*np.sqrt(np.linalg.det(lbd_w[1]))*np.prod(lbd_0)*np.exp(np.sum([-x[tr_index]*lbd_0[tr_index] for tr_index in transition_index_list])+0.5*(np.array([x-lbd_w[0]])@np.linalg.inv(lbd_w[1])@(np.array([x-lbd_w[0]])).T)[0,0])
        # if Monte Carlo, natural estimator for lbd (empirical mean and variance)
        if n_resample == 0:
            fail_samples_list = []
            for sample_index in range(N):
                if fail_times_IS[sample_index]<=t_m :
                    fail_samples_list+=[X_IS[sample_index]]
            lbd_temp[0] = np.mean(fail_samples_list,axis = 0)
            sample_std = np.std(fail_samples_list,axis = 0)
            lbd_temp[1] = np.array([[sample_std[0],0],[0,sample_std[1]]])
            del fail_samples_list

        else:
            print(f"det: {np.linalg.det(lbd[1])}")
            lbd_mu_num = 0
            lbd_sigma_num = np.zeros((2,2))
            lbd_denom = 0
            W_i_list = []
            # compute mu
            for sample_index in range(N):
                if fail_times_IS[sample_index]<=t_m :
                    W_i = W(X_IS[sample_index],lbd)
                    W_i_list+=[W_i]
                    lbd_mu_num+=W_i*X_IS[sample_index]
                    lbd_denom+=W_i
            lbd_temp[0] = lbd_mu_num/lbd_denom
            print(f"lbd_num : {lbd_mu_num}, lbd_denom : {lbd_denom}")
            # compute sigma
            W_i_list_index = 0
            for sample_index in range(N):
                if fail_times_IS[sample_index]<=t_m :
                    W_i = W_i_list[W_i_list_index]
                    W_i_list_index+=1
                    X_norm = np.array([X_IS[sample_index]-lbd_temp[0]])
                    lbd_sigma_num+=W_i*X_norm.T@X_norm
            # smoothing
            lbd_temp[1] = smoothing_parameter*(lbd_sigma_num/lbd_denom)+(1-smoothing_parameter)*lbd[1]
            del W_i_list
            gc.collect()
        lbd[:] = lbd_temp[:]
        # estimate probability and cv
        W_array = np.zeros(N)
        for sample_index in range(N):
            if fail_times_IS[sample_index]<=T :
                W_array[sample_index]=W(X_IS[sample_index],lbd)
        p_T = np.mean(W_array)
        cv = np.sqrt(np.var(W_array))/(np.sqrt(N)*p_T)
        print(f"current sampling parameters: {lbd}")
        print(f"current probability estimation: {p_T}")
        print(f"current cv estimation: {cv}")
        if t_m<=T:
            print(f"threshold reached, t_m = {t_m}")
        else:
            del W, W_array, X_IS, fail_times_IS
            gc.collect()    
        n_resample+=1
        
    # estimate probability and cv
    print("loop ended")
    print(" ")
    W_array = np.zeros(N)


    for sample_index in range(N):
        if fail_times_IS[sample_index]<=T :
            W_array[sample_index]=W(X_IS[sample_index],lbd)


    p_T = np.mean(W_array)
    if p_T>1:
        raise Exception("Final estimated probability is greater than 1")
    return p_T,cv,lbd,n_resample
def compute_pf_cv_alt(lbd_new,g,X,fail_times,obs_time):
    """
    Compute failure probability and coefficient of variation for an IS simulations until obs_time
    with a different lambda that the one used for sampling
    """
    n_run = len(X)
    # table for the computation of the multiplication factors
    f_g_array = np.zeros(n_run)
    for simu_index in range(n_run):
        # if failure before obs_time, compute f/g
        if fail_times[simu_index]<obs_time:
            f_g = 1
            for tr_index in range(len(lbd_new)):
                lbd = lbd_new[tr_index]
                x = X[simu_index][tr_index]
                f_g*= (lbd*np.exp(-lbd*x)/g[tr_index](x))
            f_g_array[simu_index] = f_g
    p_f=np.mean(f_g_array)

    # CV estimation
    cv = np.sqrt(np.var(f_g_array))/(np.sqrt(n_run)*p_f)
    return p_f,cv

def Sobol_pick_freeze_analytical(lbd_indices_v,draw_lbd,parameter_list,M):
    """
    Pick freeze with true failure probabilities in the analytical case
    """
    # draw lbd samples for pick freeze
    lbd_samples = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for m in range(M)])
    lbd_samples_v = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for m in range(M)])
    for index in lbd_indices_v:
        lbd_samples_v[:,index] = lbd_samples[:,index].copy()
    full_lbd_samples = np.concatenate((lbd_samples,lbd_samples_v))

    p_f = np.array([true_failure_proba(*lbd_sample) for lbd_sample in full_lbd_samples])

    S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
    return S

def Sobol(lbd_g,lbd_indices_v,draw_lbd,parameter_list,T,N,M):
    """
    Sobol indices of lambda of indices lbd_indices_v, with lamba drawn according to draw_lbd
    Initial sampling density lbd_g obtained for a lbd_0
    """
    # initial lbd for the sampling density
    #_,_,lbd_g,_ = AIS_CE(lbd_0,parameter_list,obs_name,T,max_time,N,alpha,[i for i in range(len(parameter_list))])
    draw_g = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd_g]
    g = [(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g]
    # draw samples according to the density
    X,Y,fail_times = sample_IS(draw_g,parameter_list,N)
    
    # draw lbd samples for pick freeze
    lbd_samples = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for m in range(M)])
    lbd_samples_v = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for m in range(M)])
    for index in lbd_indices_v:
        lbd_samples_v[:,index] = lbd_samples[:,index].copy()
    full_lbd_samples = np.concatenate((lbd_samples,lbd_samples_v))

    p_f_and_cv = np.array([compute_pf_cv_alt(lbd_sample,g,X,fail_times,T) for lbd_sample in full_lbd_samples])
    p_f = p_f_and_cv[:,0]
    cv = p_f_and_cv[:,1]
    """
    cv_max = 1
    while cv_max>eps :
        # compute p_T estimations all lbd samples
        p_f_and_cv = [compute_pf_cv_alt(lbd_sample,g,X,fail_times,T,N) for lbd_sample in full_lbd_samples]
        p_f = p_f_and_cv[:,0]
        cv = p_f_and_cv[:,1]
        cv_arg_max = np.argmax(cv[:,1])
        cv_max = cv[cv_arg_max]
        lbd_max = full_lbd_samples[cv_arg_max]
        if cv_max > eps:
            _,_,lbd = ...
    """
    S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)

    return S,np.max(cv)
def Total_Sobol(lbd_g,lbd_index,draw_lbd,parameter_list,T,N,M):
    # get all indices different from lbd_index 
    lbd_v_indices = [i for i in range(len(lbd_g)) if i!=lbd_index]
    return 1-Sobol(lbd_g,lbd_v_indices,draw_lbd,parameter_list,T,N,M)

def compute_pf_cv_mult_old(lbd_new,g_list,X,fail_times,obs_time):
    """
    Compute failure probability and coefficient of variation for an IS simulations until obs_time
    with a different lambda that the one used for sampling
    with a list of sampling density for multiple IS
    """
    n_run = len(X)
    # table for the computation of the multiplication factors
    fail_times_index = 0
    f_g_array = np.zeros(n_run)
    for simu_index in range(n_run):
        # if failure before obs_time, compute f/g    
        if fail_times[simu_index]<obs_time:
            # compute f
            f = 1
            for tr_index in range(len(lbd_new)):
                lbd = lbd_new[tr_index]
                x = X[simu_index][tr_index]
                f*= lbd*np.exp(-lbd*x)
            # compute g 
            g_mIS  = 0
            for g in g_list:
                g_part = 1
                for tr_index in range(len(lbd_new)):
                    x = X[simu_index][tr_index]
                    g_part*= g[tr_index](x)
                g_mIS+=g_part
            f_g_array[simu_index] = len(g_list)*f/g_mIS
            if f_g_array[simu_index] == 0.:
                print(f"f/g is null!\nf = lbd*np.exp(-lbd*x) = {lbd*np.exp(-lbd*x)}\ng = {g[tr_index](x)}")
            fail_times_index += 1
    p_f=np.mean(f_g_array)

    # CV estimation
    cv = np.sqrt(np.var(f_g_array))/(np.sqrt(n_run)*p_f)
    return p_f,cv

def compute_pf_cv_mult(lbd_new,g_list,X,fail_times,obs_time,g_sum_array,f_array):
    """
    Compute failure probability and coefficient of variation for an IS simulations until obs_time
    with a different lambda that the one used for sampling
    with a list of sampling density for multiple IS
    """
    n_run = len(X)
    n_run_previous = len(f_array)
    g_sum_array_len = len(g_sum_array)
    # table for the computation of the multiplication factors
    f_g_array = np.zeros(n_run)
    # pad g and f arrays for new values
    g_sum_array = np.pad(g_sum_array,(0,n_run-n_run_previous))
    f_array = np.pad(f_array,(0,n_run-n_run_previous))

    
    for simu_index in range(n_run):
        # if failure before obs_time, compute f/g    
        if fail_times[simu_index]<obs_time:
            if simu_index<n_run_previous:
                
                # compute last g sum term for mIS
                # if all g already computed
                if g_sum_array_len==n_run:
                    g_mIS = g_sum_array[simu_index]
                else:
                    g_part = 1
                    for tr_index in range(len(lbd_new)):
                        x = X[simu_index][tr_index]
                        g_part*= g_list[-1][tr_index](x)
                    g_sum_array[simu_index]+=g_part
                if g_mIS == 0.:
                    print("g_mIS is null")
                    print(f"n_run: {n_run}, g_sum_array_len: {g_sum_array_len}")
                    print("simu_index<n_run_previous")
                f_g_array[simu_index] = len(g_list)*f_array[simu_index]/g_mIS
            else:
                # compute f
                f = 1
                for tr_index in range(len(lbd_new)):
                    lbd = lbd_new[tr_index]
                    x = X[simu_index][tr_index]
                    f*= lbd*np.exp(-lbd*x)
                f_array[simu_index] = f
                # compute g 
                # if all g already computed
                if g_sum_array_len==n_run:
                    g_mIS = g_sum_array[simu_index]
                else :
                    g_mIS  = 0
                    for g in g_list:
                        g_part = 1
                        for tr_index in range(len(lbd_new)):
                            x = X[simu_index][tr_index]
                            g_part*= g[tr_index](x)
                        g_mIS+=g_part
                if g_mIS == 0.:
                    print("g_mIS is null")
                    print(f"n_run: {n_run}, g_sum_array_len: {g_sum_array_len}")
                    print("simu_index>n_run_previous")
                    for g in g_list:
                        for tr_index in range(len(lbd_new)):
                            x = X[simu_index][tr_index]  
                            print(f"g value: {g[tr_index](x)}")
                g_sum_array[simu_index] = g_mIS
                f_g_array[simu_index] = len(g_list)*f/g_mIS
                if f_g_array[simu_index] == 0.:
                    print(f"f/g is null!\nf = lbd*np.exp(-lbd*x) = {lbd*np.exp(-lbd*x)}\ng = {g[tr_index](x)}")
    p_f=np.mean(f_g_array)

    # CV estimation
    cv = np.sqrt(np.var(f_g_array))/(np.sqrt(n_run)*p_f)
    return p_f,cv

def best_combination_estimation_old(cv_list,p_f_previous,cv_previous,f_array,g_array_list,g_list_duplicates,N):
    """
    Find combination of samples that yields the estimation of p_f for a given lbd with the smallest cv
    f_g_array array of the f/g coefficient for the lbd of interest
    """
    n_total_sample = len(f_array)
    # estimation for last sample only
    f_g = f_array[n_total_sample-N:]/g_array_list[-1][n_total_sample-N:]
    p_f_last = np.mean(f_g)
    var = np.var(f_g)
    cv_last = np.sqrt(var)/(np.sqrt(N)*p_f_last)
    cv_list += [cv_last]

    # add sample for estimation only if cv decreases
    useful_g_index = np.argmin(cv_list)
    useful_X_index = np.array([0]*(N*len(cv_list)))
    useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #useful_X_index = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    current_cv = cv_list[useful_g_index]
    
    n_duplicates_used = np.array([0]*len(g_list_duplicates))

    g = g_array_list[useful_g_index].copy()
    g_candidate = g.copy()
    #print(cv_list)
    #print(useful_g_index)
    #print(f"len of f_array: {len(f_array)}")
    #print(f"len of f_array slice: {len(f_array[useful_g_index*N:(useful_g_index+1)*N])}")
    #print(f"len of g: {len(g)}")
    p_f = np.mean(f_array[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N])

    # go through increasing cv
    argsorted_cv_list = np.argsort(cv_list)
    kept_cv_indices_increasing = [0]
    cv_order_index = 1
    used_indices = [useful_g_index]
    #candidate_X_indices = useful_X_index.copy()
    candidate_X_indices = np.array([0]*(N*len(cv_list)))
    candidate_X_indices[0:N] = useful_X_index[0:N]
    
    while cv_order_index<len(cv_list):
        # index of cv in the order of the algorithm
        index = argsorted_cv_list[cv_order_index]
        n_used_indices = len(used_indices)+np.sum(n_duplicates_used)

        #additional_X_index = np.arange(index*N,(index+1)*N)
        g_additional = g_array_list[index]
        
        # candidate mIS g
        g_candidate = n_used_indices/(n_used_indices+1)*g+1/(n_used_indices+1)*g_additional

        #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
        candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = np.arange(index*N,(index+1)*N)

        f_g = f_array[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]

        p_f_test = np.mean(f_g)
        # compute variance of mIS estimator
        #var = (current_var*(n_used_indices)**2/(n_used_indices+1)**2)+np.var(f_array[index*N:(index+1)*N]/g_candidate[index*N:(index+1)*N])/(N*(n_used_indices+1)**2)
        var = 0
        for i in used_indices+[index]:
            if g_list_duplicates[i] == 0:
                var+=np.var(f_array[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
            # if 
            else:
                var+=((n_duplicates_used[i]+1)**2*np.var(f_array[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N]))

        var/=((n_used_indices+1)**2**N)
        cv = np.sqrt(var)/p_f_test

        if cv<current_cv:
            useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
            g = g_candidate.copy()
            current_cv = cv
            p_f = p_f_test
            kept_cv_indices_increasing+=[cv_order_index]

            # handle duplicated samples


            if g_list_duplicates[index]!=0:
                n_used_indices+=1
                # loop on all duplicates for this index, break loop if duplicate no longer useful
                for _ in range(g_list_duplicates[index]):
                    # candidate mIS g
                    g_candidate = n_used_indices/(n_used_indices+1)*g+1/(n_used_indices+1)*g_additional

                    #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
                    candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[(n_used_indices-1)*N:n_used_indices*N]

                    f_g = f_array[candidate_X_indices][:(n_used_indices+1)*N]/g_candidate[candidate_X_indices][:(n_used_indices+1)*N]

                    p_f_test = np.mean(f_g)
                    
                    # compute variance of mIS estimator
                    var = 0
                    for i in used_indices:
                        if g_list_duplicates[i] == 0:
                            var+=np.var(f_array[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
                        else:
                            var+=((n_duplicates_used[i]+1)**2*np.var(f_array[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N]))

                    var+=((n_duplicates_used[index]+2)**2*np.var(f_array[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N]))
                    var/=((n_used_indices+1)**2*N)
                    cv = np.sqrt(var)/p_f_test
                    if cv<current_cv:
                        n_duplicates_used[index]+=1
                        #useful_X_index = candidate_X_indices.copy()
                        useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
                        g = g_candidate.copy()
                        current_cv = cv
                        p_f = p_f_test
                        n_used_indices+=1
                        #kept_cv_indices_increasing+=[cv_order_index]
                        #cv_order_index+=1
                    else:
                        break
            used_indices += [index]
        cv_order_index+=(g_list_duplicates[index]+1)
        
    #if len(kept_cv_indices_increasing)>=2 :
        #if kept_cv_indices_increasing[-1]-kept_cv_indices_increasing[-2]!=1:
            #print(f"\n non increasing consecutive cv kept : {kept_cv_indices_increasing}")
    del f_g, p_f_last, var, cv_last, useful_X_index, g, candidate_X_indices, g_candidate,n_duplicates_used
    gc.collect() 
    if current_cv<cv_previous:
        return p_f,current_cv,False
    return p_f_previous,cv_previous,True

def best_combination_estimation_simplified(cv_list,p_f_previous,cv_previous,f_array,g_array_list,g_list_duplicates,N):
    """
    Find combination of samples that yields the estimation of p_f for a given lbd with the smallest cv
    f_g_array array of the f/g coefficient for the lbd of interest
    """
    n_total_sample = len(f_array)
    # estimation for last sample only
    f_g = f_array[n_total_sample-N:]/g_array_list[-1][n_total_sample-N:]
    p_f_last = np.mean(f_g)
    var = np.var(f_g)
    cv_last = np.sqrt(var)/(np.sqrt(N)*p_f_last)
    cv_list += [cv_last]
    # add sample for estimation only if cv decreases
    useful_g_index = np.argmin(cv_list)
    useful_X_index = np.zeros(N * len(cv_list), dtype=int)
    useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #useful_X_index = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    current_cv = cv_list[useful_g_index]
    
    print(f"cv_list : {cv_list}")
    print(f"useful_g_index: {useful_g_index}")
    #g = g_array_list[useful_g_index].copy()
    #g_candidate = g.copy()
    print(f"len f_array: {len(f_array)}")
    print(f"len g_array_list[useful_g_index]: {len(g_array_list[useful_g_index])}")

    p_f = np.mean(f_array[useful_g_index*N:(useful_g_index+1)*N]/g_array_list[useful_g_index][useful_g_index*N:(useful_g_index+1)*N])

    # go through increasing cv
    argsorted_cv_list = np.argsort(cv_list)
    #kept_cv_indices_increasing = [0]
    cv_order_index = 1
    used_indices = [useful_g_index]
    #candidate_X_indices = useful_X_index.copy()
    candidate_X_indices = np.zeros(N * len(cv_list), dtype=int)
    candidate_X_indices[0:N] = useful_X_index[0:N]
    
    g_candidate = np.zeros_like(g_array_list[0])
    g = np.zeros_like(g_array_list[0])
    g[:] = g_array_list[useful_g_index][:]
    f_g = np.zeros_like(g_array_list[0])

    while cv_order_index<len(cv_list):
        # index of cv in the order of the algorithm
        index = argsorted_cv_list[cv_order_index]
        n_used_indices = len(used_indices)

        #additional_X_index = np.arange(index*N,(index+1)*N)
        #g_additional = g_array_list[index]
        
        # candidate mIS g
        #np.add(g * (n_used_indices / (n_used_indices + 1)), g_array_list[index] * (1 / (n_used_indices + 1)), out=g_candidate)
        #print(used_indices+[index])
        g_candidate[:] = np.sum([g_array_list[i] for i in used_indices + [index]], axis=0)[:] / (n_used_indices + 1)

        #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
        candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = np.arange(index*N,(index+1)*N)

        f_g = f_array[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        np.divide(f_array[candidate_X_indices[:(n_used_indices+1)*N]], 
          g_candidate[candidate_X_indices[:(n_used_indices+1)*N]], 
          out=f_g[:(n_used_indices+1)*N])
        
        p_f_test = np.mean(f_g[:(n_used_indices+1)*N])
        # compute variance of mIS estimator
        #var = (current_var*(n_used_indices)**2/(n_used_indices+1)**2)+np.var(f_array[index*N:(index+1)*N]/g_candidate[index*N:(index+1)*N])/(N*(n_used_indices+1)**2)
        var = np.var(f_g[:(n_used_indices+1)*N])
        var/=((n_used_indices+1)**2*N)
        cv = np.sqrt(var)/p_f_test

        if cv<current_cv:
            useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
            g[:] = g_candidate[:]
            current_cv = cv
            p_f = p_f_test
            #kept_cv_indices_increasing+=[cv_order_index]
            used_indices += [index]
        cv_order_index+=(g_list_duplicates[index]+1)

    #if len(kept_cv_indices_increasing)>=2 :
        #if kept_cv_indices_increasing[-1]-kept_cv_indices_increasing[-2]!=1:
            #print(f"\n non increasing consecutive cv kept : {kept_cv_indices_increasing}")
    del f_g, var, cv_last, useful_X_index, g, candidate_X_indices, g_candidate,argsorted_cv_list,used_indices
    #gc.collect() 
    #del useful_X_index, argsorted_cv_list,kept_cv_indices_increasing,used_indices,f_g, g, candidate_X_indices
    gc.collect() 
    if current_cv<cv_previous:
        return p_f,current_cv,False
    return p_f_previous,cv_previous,True

def best_combination_estimation(true_pf,cv_list,p_f_previous,var_previous,cv_previous,ess_previous,f_array_ind,f_array_true,g_array_list,g_list_duplicates,N,recheck):
    """
    Find combination of samples that yields the estimation of p_f for a given lbd with the smallest cv
    f_g_array array of the f/g coefficient for the lbd of interest
    """
    n_total_sample = len(f_array_ind)
    # estimation for last sample only
    f_g_ind = f_array_ind[n_total_sample-N:]/g_array_list[-1][n_total_sample-N:]
    p_f_last = np.mean(f_g_ind)

    cv_list += [np.std(f_g_ind)/(np.sqrt(N)*p_f_last)]

    if not recheck:
        return p_f_previous,cv_previous,var_previous,ess_previous,True

    # add sample for estimation only if cv decreases
    useful_g_index = np.argmin(cv_list)
    useful_X_index = np.zeros(N * len(cv_list), dtype=int)
    useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #useful_X_index = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    current_cv = cv_list[useful_g_index]
    
    n_duplicates_used = np.zeros(len(g_list_duplicates), dtype=int)

    g = g_array_list[useful_g_index].copy()
    f_g_ind = f_array_ind[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    f_g_true = f_array_true[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    current_var = np.var(f_g_ind)/N
    ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)
    g_candidate = g.copy()
    p_f = np.mean(f_g_ind)

    # go through increasing cv
    argsorted_cv_list = np.argsort(cv_list)
    #kept_cv_indices_increasing = [0]
    cv_order_index = 1
    used_indices = [useful_g_index]
    #candidate_X_indices = useful_X_index.copy()
    candidate_X_indices = np.zeros(N * len(cv_list), dtype=int)
    candidate_X_indices[0:N] = useful_X_index[0:N]
    
    while cv_order_index<len(cv_list):
        # index of cv in the order of the algorithm
        index = argsorted_cv_list[cv_order_index]
        n_used_indices = len(used_indices)+np.sum(n_duplicates_used)

        #additional_X_index = np.arange(index*N,(index+1)*N)
        g_additional = g_array_list[index]
        
        # candidate mIS g
        #g_candidate[:] = np.sum([g_array_list[i] for i in used_indices + [index]], axis=0)[:] / (n_used_indices + 1)
        g_candidate = n_used_indices/(n_used_indices+1)*g+1/(n_used_indices+1)*g_additional

        #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
        candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = np.arange(index*N,(index+1)*N)

        f_g_ind = f_array_ind[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        f_g_true = f_array_true[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        p_f_test = np.mean(f_g_ind)
        # compute variance of mIS estimator
        #var = (current_var*(n_used_indices)**2/(n_used_indices+1)**2)+np.var(f_array[index*N:(index+1)*N]/g_candidate[index*N:(index+1)*N])/(N*(n_used_indices+1)**2)
        var = 0
        for i in used_indices+[index]:
            if g_list_duplicates[i] == 0:
                var+=np.var(f_array_ind[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
            # if 
            else:
                var+=(1+n_duplicates_used[i])*np.var(f_array_ind[i*N:(i+1+n_duplicates_used[i])*N]/g_candidate[i*N:(i+1+n_duplicates_used[i])*N])

        var/=((n_used_indices+1)**2*N)
        cv = np.sqrt(var)/p_f_test

        if cv<current_cv:
            useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
            g = g_candidate.copy()
            current_cv = cv
            current_var = var
            p_f = p_f_test
            ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)
            #kept_cv_indices_increasing+=[cv_order_index]

            # handle duplicated samples


            if g_list_duplicates[index]!=0:
                n_used_indices+=1
                # loop on all duplicates for this index, break loop if duplicate no longer useful
                for _ in range(g_list_duplicates[index]):
                    # candidate mIS g
                    g_candidate = n_used_indices/(n_used_indices+1)*g+1/(n_used_indices+1)*g_additional

                    #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
                    # if (n_used_indices+1)*N>len(candidate_X_indices)-1:
                    #     print("Problem: trying to access non existing indices for candidates")
                    #     print(f"length of candidate_X_indices: {len(candidate_X_indices)}, index tried: {(n_used_indices+1)*N}, n_used_indices: {n_used_indices}")
                    candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[(n_used_indices-1)*N:n_used_indices*N]

                    f_g_ind = f_array_ind[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
                    f_g_true = f_array_true[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
                    p_f_test = np.mean(f_g_ind)
                    
                    # compute variance of mIS estimator
                    var = 0
                    for i in used_indices:
                        if g_list_duplicates[i] == 0:
                            var+=np.var(f_array_ind[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
                        else:
                            var+=(1+n_duplicates_used[i])*np.var(f_array_ind[i*N:(i+1+n_duplicates_used[i])*N]/g_candidate[i*N:(i+1+n_duplicates_used[i])*N])

                    var+=(2+n_duplicates_used[index])*np.var(f_array_ind[i*N:(i+2+n_duplicates_used[index])*N]/g_candidate[i*N:(i+2+n_duplicates_used[index])*N])
                    var/=((n_used_indices+n_duplicates_used[index]+1)**2*N)
                    cv = np.sqrt(var)/p_f_test
                    if cv<current_cv:
                        n_duplicates_used[index]+=1
                        #useful_X_index = candidate_X_indices.copy()
                        useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
                        g = g_candidate.copy()
                        current_cv = cv
                        current_var = var
                        p_f = p_f_test
                        ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)
                        n_used_indices+=1
                        #kept_cv_indices_increasing+=[cv_order_index]
                        #cv_order_index+=1
                    else:
                        break
            used_indices += [index]
        cv_order_index+=(g_list_duplicates[index]+1)
        
    #if len(kept_cv_indices_increasing)>=2 :
        #if kept_cv_indices_increasing[-1]-kept_cv_indices_increasing[-2]!=1:
            #print(f"\n non increasing consecutive cv kept : {kept_cv_indices_increasing}")
    #del f_g, p_f_last, var, cv_last, useful_X_index, g, candidate_X_indices, g_candidate,n_duplicates_used
    #gc.collect() 
    if current_cv<cv_previous:
        return p_f,current_cv,current_var,ESS_estim,False

    return p_f_previous,cv_previous,var_previous,ess_previous,True

def best_combination_estimation_simpler(true_pf,cv_list,p_f_previous,var_previous,cv_previous,ess_previous,f_array_ind,f_array_true,g_array_list,g_list_duplicates,N):
    """
    Find combination of samples that yields the estimation of p_f for a given lbd with the smallest cv
    f_g_array array of the f/g coefficient for the lbd of interest
    """
    # f_array = indicator function of failure zone*f_array_true
    n_total_sample = len(f_array_true)
    # estimation for last sample only
    f_g_ind = f_array_ind[n_total_sample-N:]/g_array_list[-1][n_total_sample-N:]
    p_f_last = np.mean(f_g_ind)
    #cv_last = np.sqrt(var)/(np.sqrt(N)*p_f_last)
    #var_list += [np.var(f_g)/N]
    cv_list += [np.std(f_g_ind)/(np.sqrt(N)*p_f_last)]
    #var_list += [np.sqrt(np.var(f_g)/N)/p_f_last]
    # add sample for estimation only if cv decreases
    useful_g_index = np.argmin(cv_list)
    useful_X_index = np.zeros(N * len(cv_list), dtype=int)
    useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #useful_X_index = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #current_var = var_list[-1]
    #current_cv = np.sqrt(var_list[-1])/p_f_last
    current_cv = cv_list[useful_g_index]

    
    #n_duplicates_used = np.zeros(len(g_list_duplicates), dtype=int)

    g = g_array_list[useful_g_index].copy()
    f_g_ind = f_array_ind[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    f_g_true = f_array_true[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    current_var = np.var(f_g_ind)/N
    ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)
    g_candidate = g.copy()
    #print(cv_list)
    #print(useful_g_index)
    #print(f"len of f_array: {len(f_array)}")
    #print(f"len of f_array slice: {len(f_array[useful_g_index*N:(useful_g_index+1)*N])}")
    #print(f"len of g: {len(g)}")
    p_f = np.mean(f_g_ind)

    # go through increasing cv
    #argsorted_var_list = np.argsort(var_list)
    argsorted_cv_list = np.argsort(cv_list)
    #kept_cv_indices_increasing = [0]
    cv_order_index = 1
    used_indices = [useful_g_index]
    #candidate_X_indices = useful_X_index.copy()
    candidate_X_indices = np.zeros(N * len(cv_list), dtype=int)
    candidate_X_indices[0:N] = useful_X_index[0:N]
    # for l in itertools.permutations(argsorted_var_list):
    #     cv_order_index = 1
    #     useful_g_index = l[0]
    #     useful_X_index = np.zeros(N * len(var_list), dtype=int)
    #     useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #     used_indices = [useful_g_index]
    #     p_f = np.mean(f_array[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N])
    #     candidate_X_indices = np.zeros(N * len(var_list), dtype=int)
    #     candidate_X_indices[0:N] = useful_X_index[0:N]
    while cv_order_index<len(cv_list):
        #candidate_X_indices[:] = useful_X_index[:]
        # index of cv in the order of the algorithm
        index = argsorted_cv_list[cv_order_index]
        #index = l[cv_order_index]
        n_used_indices = len(used_indices)

        #additional_X_index = np.arange(index*N,(index+1)*N)
        g_additional = g_array_list[index]
        
        # candidate mIS g
        g_candidate = n_used_indices/(n_used_indices+1)*g+1/(n_used_indices+1)*g_additional
        
        #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
        candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = np.arange(index*N,(index+1)*N)

        f_g_ind = f_array_ind[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        f_g_true = f_array_true[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        p_f_test = np.mean(f_g_ind)
        # compute variance of mIS estimator
        #var = (current_var*(n_used_indices)**2/(n_used_indices+1)**2)+np.var(f_array[index*N:(index+1)*N]/g_candidate[index*N:(index+1)*N])/(N*(n_used_indices+1)**2)
        var = 0
        for i in used_indices+[index]:
            var+=np.var(f_array_ind[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
            
        var/=((n_used_indices+1)**2*N)
        cv = np.sqrt(var)/p_f_test

        #if var<current_var:
        if cv<current_cv:
            #useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
            g = g_candidate.copy()
            current_cv = cv
            current_var = var
            p_f = p_f_test 
            ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)
            #kept_cv_indices_increasing+=[cv_order_index]
            # handle duplicated samples
            used_indices += [index]
        cv_order_index+=1
        
    #if len(kept_cv_indices_increasing)>=2 :
        #if kept_cv_indices_increasing[-1]-kept_cv_indices_increasing[-2]!=1:
            #print(f"\n non increasing consecutive cv kept : {kept_cv_indices_increasing}")
    #del f_g, p_f_last, var, cv_last, useful_X_index, g, candidate_X_indices, g_candidate,n_duplicates_used
    #gc.collect() 
    #if current_var<var_previous:
    if current_cv<cv_previous:
        return p_f,current_cv,current_var,ESS_estim,False
    # else:
        #if np.abs(current_var-var_previous)/np.min([current_var,var_previous])>var_previous:
        # if np.abs(current_cv-cv_previous)/np.min([current_cv,cv_previous])>cv_previous:
        #     print("PROBLEM: new computed var superior to previous cv")
        #     print(f"previous cv : {cv_previous}, current cv : {current_cv}")
        #     print(f"previous var : {var_previous}, current var : {current_var}")
    return p_f_previous,cv_previous,var_previous,ess_previous,True

def best_combination_estimation_ideal(true_pf,error_list,p_f_previous,var_previous,error_previous,ess_previous,f_array_ind,f_array_true,g_array_list,g_list_duplicates,N):
    """
    Find combination of samples that yields the estimation of p_f for a given lbd with the smallest cv
    f_g_array array of the f/g coefficient for the lbd of interest
    """
    n_total_sample = len(f_array_ind)
    # estimation for last sample only
    f_g_ind = f_array_ind[n_total_sample-N:]/g_array_list[-1][n_total_sample-N:]
    p_f_last = np.mean(f_g_ind)
    #cv_last = np.sqrt(var)/(np.sqrt(N)*p_f_last)
    #var_list += [np.var(f_g)/N]
    #cv_list += [np.std(f_g)/(np.sqrt(N)*p_f_last)]
    error_list += [np.abs(p_f_last/true_pf-1)]
    #print(f"errors list: {error_list}")
    #var_list += [np.sqrt(np.var(f_g)/N)/p_f_last]
    # add sample for estimation only if cv decreases
    useful_g_index = np.argmin(error_list)
    useful_X_index = np.zeros(N * len(error_list), dtype=int)
    useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #useful_X_index = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #current_var = var_list[-1]
    #current_cv = np.sqrt(var_list[-1])/p_f_last
    current_error = error_list[useful_g_index]

    

    g = g_array_list[useful_g_index].copy()
    f_g_ind = f_array_ind[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    f_g_true = f_array_true[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]    
    current_var = np.var(f_g_ind)/N
    ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)
    g_candidate = g.copy()
    #print(cv_list)
    #print(useful_g_index)
    #print(f"len of f_array: {len(f_array)}")
    #print(f"len of f_array slice: {len(f_array[useful_g_index*N:(useful_g_index+1)*N])}")
    #print(f"len of g: {len(g)}")
    p_f = np.mean(f_g_ind)

    # go through increasing cv
    #argsorted_var_list = np.argsort(var_list)
    argsorted_error_list = np.argsort(error_list)
    #kept_cv_indices_increasing = [0]
    # go through each permuation of sample batch
    cv_order_index = 1
    used_indices = [useful_g_index]
    #candidate_X_indices = useful_X_index.copy()
    candidate_X_indices = np.zeros(N * len(error_list), dtype=int)
    candidate_X_indices[0:N] = useful_X_index[0:N]


    while cv_order_index<len(error_list):
        #candidate_X_indices[:] = useful_X_index[:]
        # index of cv in the order of the algorithm
        index = argsorted_error_list[cv_order_index]
        #index = l[cv_order_index]
        n_used_indices = len(used_indices)

        #additional_X_index = np.arange(index*N,(index+1)*N)
        g_additional = g_array_list[index]
        
        # candidate mIS g
        g_candidate = n_used_indices/(n_used_indices+1)*g+1/(n_used_indices+1)*g_additional

        #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
        candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = np.arange(index*N,(index+1)*N)

        f_g_ind = f_array_ind[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        f_g_true = f_array_true[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        p_f_test = np.mean(f_g_ind)
        # compute variance of mIS estimator
        #var = (current_var*(n_used_indices)**2/(n_used_indices+1)**2)+np.var(f_array[index*N:(index+1)*N]/g_candidate[index*N:(index+1)*N])/(N*(n_used_indices+1)**2)
        var = 0
        for i in used_indices+[index]:
            var+=np.var(f_array_ind[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
            
        var/=((n_used_indices+1)**2*N)
        cv = np.sqrt(var)/p_f_test

        error =  np.abs(p_f_test/true_pf-1)

        #if var<current_var:
        if error<current_error:
            #useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
            g = g_candidate.copy()
            current_cv = cv
            current_var = var
            current_error = error
            p_f = p_f_test
            ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)
            
            #kept_cv_indices_increasing+=[cv_order_index]
            
            # handle duplicated samples
            used_indices += [index]
        cv_order_index+=1
        
    #if len(kept_cv_indices_increasing)>=2 :
        #if kept_cv_indices_increasing[-1]-kept_cv_indices_increasing[-2]!=1:
            #print(f"\n non increasing consecutive cv kept : {kept_cv_indices_increasing}")
    #del f_g, p_f_last, var, cv_last, useful_X_index, g, candidate_X_indices, g_candidate,n_duplicates_used
    #gc.collect() 
    #if current_var<var_previous:
    if current_error<error_previous:
        return p_f,current_error,current_var,ESS_estim,False
    elif current_error>error_previous:
        for sample_batch_list_permutation in list(itertools.permutations(argsorted_error_list))[1:]:
            useful_g_index = sample_batch_list_permutation[0]
            useful_X_index = np.zeros(N * len(error_list), dtype=int)
            useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)

            cv_order_index = 1
            used_indices = [useful_g_index]
            #candidate_X_indices = useful_X_index.copy()
            candidate_X_indices = np.zeros(N * len(error_list), dtype=int)
            candidate_X_indices[0:N] = useful_X_index[0:N]


            while cv_order_index<len(error_list):
                #candidate_X_indices[:] = useful_X_index[:]
                # index of cv in the order of the algorithm
                index = argsorted_error_list[cv_order_index]
                #index = l[cv_order_index]
                n_used_indices = len(used_indices)

                #additional_X_index = np.arange(index*N,(index+1)*N)
                g_additional = g_array_list[index]
                
                # candidate mIS g
                g_candidate = n_used_indices/(n_used_indices+1)*g+1/(n_used_indices+1)*g_additional

                #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
                candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = np.arange(index*N,(index+1)*N)

                f_g_ind = f_array_ind[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
                f_g_true = f_array_true[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
                p_f_test = np.mean(f_g_ind)
                # compute variance of mIS estimator
                #var = (current_var*(n_used_indices)**2/(n_used_indices+1)**2)+np.var(f_array[index*N:(index+1)*N]/g_candidate[index*N:(index+1)*N])/(N*(n_used_indices+1)**2)
                var = 0
                for i in used_indices+[index]:
                    var+=np.var(f_array_ind[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
                    
                var/=((n_used_indices+1)**2*N)
                cv = np.sqrt(var)/p_f_test

                error =  np.abs(p_f_test/true_pf-1)

                #if var<current_var:
                if error<current_error:

                    #useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
                    g = g_candidate.copy()
                    current_cv = cv
                    current_var = var
                    current_error = error
                    p_f = p_f_test
                    ESS_estim = np.sum(f_g_true)**2/np.sum(f_g_true**2)
                    
                    #kept_cv_indices_increasing+=[cv_order_index]
                    
                    # handle duplicated samples
                    used_indices += [index]
                cv_order_index+=1
            if  current_error<error_previous:
                return p_f,current_error,current_var,ESS_estim,False
        #if np.abs(current_error-error_previous)/np.min([current_error,error_previous])>error_previous:
        if current_error>error_previous:
            print("PROBLEM: new computed error superior to previous error")
            #print(f"previous cv : {cv_previous}, current cv : {current_cv}")
            print(f"previous error : {error_previous}, current error : {current_error}")
           # print(f"previous var : {var_previous}, current var : {current_var}")
    return p_f_previous,error_previous,var_previous,ess_previous,True

def best_combination_estimation_ESS(true_pf,error_list,p_f_previous,var_previous,error_previous,f_array_ind,f_array_true,g_array_list,g_list_duplicates,N):
    """
    Find combination of samples that yields the estimation of p_f for a given lbd with the smallest cv
    f_g_array array of the f/g coefficient for the lbd of interest
    """
    n_total_sample = len(f_array_true)
    # estimation for last sample only
    f_g_true = f_array_true[n_total_sample-N:]/g_array_list[-1][n_total_sample-N:]

    error_list += [np.sum(f_g_true)**2/np.sum(f_g_true**2)]

    # add sample for estimation only if cv decreases
    useful_g_index = np.argmax(error_list)
    useful_X_index = np.zeros(N * len(error_list), dtype=int)
    useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    current_error = error_list[useful_g_index]
    
    

    g = g_array_list[useful_g_index].copy()
    f_g_ind = f_array_ind[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    f_g_true = f_array_true[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    current_var = np.var(f_g_ind)/N
    g_candidate = g.copy()
    #print(cv_list)
    #print(useful_g_index)
    #print(f"len of f_array: {len(f_array)}")
    #print(f"len of f_array slice: {len(f_array[useful_g_index*N:(useful_g_index+1)*N])}")
    #print(f"len of g: {len(g)}")
    p_f = np.mean(f_g_ind)

    # go through increasing cv
    #argsorted_var_list = np.argsort(var_list)
    argsorted_error_list = np.argsort(error_list)[::-1]
    #kept_cv_indices_increasing = [0]
    cv_order_index = 1
    used_indices = [useful_g_index]
    #candidate_X_indices = useful_X_index.copy()
    candidate_X_indices = np.zeros(N * len(error_list), dtype=int)
    candidate_X_indices[0:N] = useful_X_index[0:N]

    while cv_order_index<len(error_list):
        #candidate_X_indices[:] = useful_X_index[:]
        # index of cv in the order of the algorithm
        index = argsorted_error_list[cv_order_index]
        #index = l[cv_order_index]
        n_used_indices = len(used_indices)

        #additional_X_index = np.arange(index*N,(index+1)*N)
        g_additional = g_array_list[index]
        
        # candidate mIS g
        g_candidate = n_used_indices/(n_used_indices+1)*g+1/(n_used_indices+1)*g_additional

        #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
        candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = np.arange(index*N,(index+1)*N)

        f_g_ind = f_array_ind[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        f_g_true = f_array_true[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        p_f_test = np.mean(f_g_ind)
        # compute variance of mIS estimator
        #var = (current_var*(n_used_indices)**2/(n_used_indices+1)**2)+np.var(f_array[index*N:(index+1)*N]/g_candidate[index*N:(index+1)*N])/(N*(n_used_indices+1)**2)
        var = 0
        for i in used_indices+[index]:
            var+=np.var(f_array_ind[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
            
        var/=((n_used_indices+1)**2*N)
        cv = np.sqrt(var)/p_f_test

        error =  np.sum(f_g_true)**2/np.sum(f_g_true**2)

        #if var<current_var:
        if error>current_error:
            #useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
            g = g_candidate.copy()
            current_cv = cv
            current_var = var
            current_error = error
            p_f = p_f_test
            #kept_cv_indices_increasing+=[cv_order_index]

            # handle duplicated samples
            used_indices += [index]
        cv_order_index+=1
        
    #if len(kept_cv_indices_increasing)>=2 :
        #if kept_cv_indices_increasing[-1]-kept_cv_indices_increasing[-2]!=1:
            #print(f"\n non increasing consecutive cv kept : {kept_cv_indices_increasing}")
    #del f_g, p_f_last, var, cv_last, useful_X_index, g, candidate_X_indices, g_candidate,n_duplicates_used
    #gc.collect() 

    # print("\nf/g values:")
    # print(f_g)
    # print("\n")

    #if current_var<var_previous:
    if current_error>error_previous:
        return p_f,current_error,current_var,False
    else:
        #if np.abs(current_var-var_previous)/np.min([current_var,var_previous])<var_previous:
        # if current_error<error_previous:
            # print("PROBLEM: new computed ESS inferior to previous ESS")
            # #print(f"previous cv : {cv_previous}, current cv : {current_cv}")
            # print(f"previous error : {error_previous}, current error : {current_error}")
            # print(f"previous var : {var_previous}, current var : {current_var}")

        return p_f_previous,error_previous,var_previous,True

def best_combination_estimation_brute(true_pf,error_list,p_f_previous,var_previous,error_previous,ess_previous,f_array_ind,f_array_true,g_array_list,g_list_duplicates,N):
    """
    Find combination of samples that yields the estimation of p_f for a given lbd with the smallest cv
    f_g_array array of the f/g coefficient for the lbd of interest
    """
    n_total_sample = len(f_array_ind)
    # estimation for last sample only
    f_g_ind = f_array_ind[n_total_sample-N:]/g_array_list[-1][n_total_sample-N:]
    p_f_last = np.mean(f_g_ind)
    #cv_last = np.sqrt(var)/(np.sqrt(N)*p_f_last)
    #var_list += [np.var(f_g)/N]
    #cv_list += [np.std(f_g)/(np.sqrt(N)*p_f_last)]
    error_list += [np.abs(p_f_last/true_pf-1)]
    #var_list += [np.sqrt(np.var(f_g)/N)/p_f_last]
    # add sample for estimation only if cv decreases
    useful_g_index = np.argmin(error_list)
    useful_X_index = np.zeros(N * len(error_list), dtype=int)
    useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #useful_X_index = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #current_var = var_list[-1]
    #current_cv = np.sqrt(var_list[-1])/p_f_last
    current_error = error_list[useful_g_index]

    

    g = g_array_list[useful_g_index].copy()
    f_g_ind = f_array_ind[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    f_g_true = f_array_true[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]    
    current_var = np.var(f_g_ind)/N
    ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)
    g_candidate = g.copy()
    #print(cv_list)
    #print(useful_g_index)
    #print(f"len of f_array: {len(f_array)}")
    #print(f"len of f_array slice: {len(f_array[useful_g_index*N:(useful_g_index+1)*N])}")
    #print(f"len of g: {len(g)}")
    p_f = np.mean(f_g_ind)

    # go through increasing cv
    #argsorted_var_list = np.argsort(var_list)
    argsorted_error_list = np.argsort(error_list)
    #kept_cv_indices_increasing = [0]
    # go through each permuation of sample batch
    cv_order_index = 1
    used_indices = [useful_g_index]
    #candidate_X_indices = useful_X_index.copy()
    candidate_X_indices = np.zeros(N * len(error_list), dtype=int)
    candidate_X_indices[0:N] = useful_X_index[0:N]

    while cv_order_index<len(error_list):
        #candidate_X_indices[:] = useful_X_index[:]
        # index of cv in the order of the algorithm
        index = argsorted_error_list[cv_order_index]
        #index = l[cv_order_index]
        n_used_indices = len(used_indices)

        #additional_X_index = np.arange(index*N,(index+1)*N)
        g_additional = g_array_list[index]
        
        # candidate mIS g
        g_candidate = n_used_indices/(n_used_indices+1)*g+1/(n_used_indices+1)*g_additional
        #g_candidate[:] = np.sum([g_array_list[i] for i in used_indices + [index]], axis=0)[:] / (n_used_indices + 1)

        #candidate_X_indices = np.concatenate((useful_X_index,additional_X_index))
        candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N] = np.arange(index*N,(index+1)*N)

        f_g_ind = f_array_ind[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        f_g_true = f_array_true[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]
        p_f_test = np.mean(f_g_ind)
        # compute variance of mIS estimator
        #var = (current_var*(n_used_indices)**2/(n_used_indices+1)**2)+np.var(f_array[index*N:(index+1)*N]/g_candidate[index*N:(index+1)*N])/(N*(n_used_indices+1)**2)
        var = 0
        for i in used_indices+[index]:
            var+=np.var(f_array_ind[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
            
        var/=((n_used_indices+1)**2*N)
        cv = np.sqrt(var)/p_f_test

        error =  np.abs(p_f_test/true_pf-1)

        g = g_candidate.copy()
        current_cv = cv
        current_var = var
        current_error = error
        p_f = p_f_test
        ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)
        used_indices += [index]
        cv_order_index+=1


    if current_error<error_previous:
        return p_f,current_error,current_var,ESS_estim,False
    # else:
    #     #if np.abs(current_var-var_previous)/np.min([current_var,var_previous])>var_previous:
    #     if np.abs(current_error-error_previous)/np.min([current_error,error_previous])>error_previous:
    #         print("PROBLEM: new computed error superior to previous error")
    #         #print(f"previous cv : {cv_previous}, current cv : {current_cv}")
    #         print(f"previous errror : {error_previous}, current error : {current_error}")
    #         print(f"previous var : {var_previous}, current var : {current_var}")
    return p_f,current_error,current_var,ESS_estim,True

def best_combination_estimation_last(true_pf,cv_list,p_f_previous,var_previous,cv_previous,ess_previous,f_array_ind,f_array_true,g_array_list,g_list_duplicates,N):
    """
    Find combination of samples that yields the estimation of p_f for a given lbd with the smallest cv
    f_g_array array of the f/g coefficient for the lbd of interest
    """
    n_total_sample = len(f_array_ind)
    # estimation for last sample only
    f_g_ind = f_array_ind[n_total_sample-N:]/g_array_list[-1][n_total_sample-N:]
    p_f_last = np.mean(f_g_ind)

    cv_list += [np.std(f_g_ind)/(np.sqrt(N)*p_f_last)]
    # add sample for estimation only if cv decreases
    useful_g_index = len(cv_list)-1
    useful_X_index = np.zeros(N * len(cv_list), dtype=int)
    useful_X_index[0:N] = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    #useful_X_index = np.arange(useful_g_index*N,(useful_g_index+1)*N)
    current_cv = cv_list[useful_g_index]
    
    n_duplicates_used = np.zeros(len(g_list_duplicates), dtype=int)

    g = g_array_list[useful_g_index].copy()
    f_g_ind2 = f_array_ind[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    f_g_true = f_array_true[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N]
    current_var = np.var(f_g_ind2)/N
    ESS_estim =  np.sum(f_g_true)**2/np.sum(f_g_true**2)


    return p_f_last,current_cv,current_var,ESS_estim,False


def update_g_array_list(lbd_new,g_list,g_array_list,X,fail_times,obs_time,N):
    """
    Update list of g values that are necessary for computation of the estimator
    """
    n_run = len(X)
    # update previous arrays with evaluation of new X values
    for g_index in range(len(g_array_list)):
        g_array_list[g_index] = np.pad(g_array_list[g_index], (0, max(0, n_run - len(g_array_list[g_index]))), mode='constant')
        #g_array_list[g_index].resize(n_run,refcheck=False)
        # for simu_index in range(n_run-N,n_run):
        #     g_array_list[g_index][simu_index] = 1.
        #     if fail_times[simu_index]<obs_time:
        #         for tr_index in range(len(lbd_new)):
        #             x = X[simu_index][tr_index]
        #             g_array_list[g_index][simu_index]*= g_list[g_index][tr_index](x)

        # Create a mask for simulations where fail_times < obs_time
        #mask = fail_times[n_run-N:n_run] < obs_time

        # Initialize g_array to 1 for all simulations
        #g_array_list[g_index][n_run-N:n_run] = 1.
        g_array_list[g_index][n_run-N:n_run] = np.array([g_list[g_index](X[simu_index]) for simu_index in range(n_run-N,n_run)])#range(n_run-N,n_run)])
        # If the fail time is less than obs_time, calculate g_array
        # if np.any(mask):
        #     X_masked = X[n_run-N:n_run][mask]

        #     # Apply g_list functions to each transition for each simulation
        #     # g_values = np.array([[g_list[g_index][tr_index](X_masked[simu_index, tr_index])
        #     #                     for tr_index in range(len(lbd_new))]
        #     #                     for simu_index in range(X_masked.shape[0])])
        #     g_values = np.array([g_list[g_index](X_masked[simu_index]) for simu_index in range(X_masked.shape[0])])

        #     # Compute the product across transitions for each simulation
        #     #g_prod = np.prod(g_values, axis=1)

        #     # Assign computed values back to g_array_list at the corresponding index
        #     g_array_list[g_index][n_run-N:n_run][mask] = g_values#g_prod


    # array for new g
    # new_g_array = np.ones(n_run)
    # for simu_index in range(n_run):
    #     if fail_times[simu_index]<obs_time:
    #         for tr_index in range(len(lbd_new)):
    #             x = X[simu_index][tr_index]
    #             new_g_array[simu_index]*= g_list[-1][tr_index](x)

    # Create a mask for simulations where fail_times < obs_time
    #mask = fail_times < obs_time

    # Initialize new_g_array with ones for all simulations
    #new_g_array = np.ones(n_run)

    # Filter X based on the mask for fail_times < obs_time
    #X_masked = X[mask]

    # Apply g_list functions for each transition to the masked X values
    # g_values = np.array([[g_list[-1][tr_index](X_masked[simu_index, tr_index])
    #                     for tr_index in range(len(lbd_new))]
    #                     for simu_index in range(X_masked.shape[0])])
    
    new_g_array = np.array([g_list[-1](X[simu_index])
                        for simu_index in range(n_run)])
    #g_values = np.array([g_list[-1](X_masked[simu_index])
                        #for simu_index in range(X_masked.shape[0])])
    # Compute the product of g_list values across transitions for each simulation
    #g_prod = np.prod(g_values, axis=1)

    # Update new_g_array only where fail_times < obs_time
    #print(f"new_g_array[mask] : {new_g_array[mask]}")
    #print(f"g_values : {g_values}")
    #new_g_array[mask] = g_values#g_prod
    
    # add new g array
    g_array_list += [new_g_array]
    
def update_g_array_baseline(g,g_array_baseline,X):
    """
    Update list of g values that are necessary for computation of the estimator
    """
    n_run = len(X)
    n_run_previous = len(g_array_baseline)
    new_g_values = np.array([g(X[simu_index]) for simu_index in range(n_run_previous,n_run)])
    new_g_array_baseline = np.concatenate([g_array_baseline,new_g_values])
    return new_g_array_baseline

def compute_pf_cv_mult_alt(true_pf,var_list,p_f_previous,var_previous,error_previous,lbd_new,g_list_duplicates,X,fail_times,obs_time,g_array_list,f_array_list,lbd_index):
    """
    Compute failure probability and coefficient of variation for an IS simulations until obs_time
    with a different lambda that the one used for sampling
    with a list of sampling density for multiple IS
    """
    n_run = len(X)
    n_run_previous = len(f_array_list[lbd_index])

    #update f
    f_array_list[lbd_index] = np.pad(f_array_list[lbd_index],(0,n_run-n_run_previous))
    if len(f_array_list[lbd_index]) == 0:
        raise Exception("f_array empty 1")

    
    # for simu_index in range(n_run_previous,n_run):
    #     # if failure before obs_time
    #     if fail_times[simu_index]<obs_time:
    #         # compute f
    #         f = 1
    #         for tr_index in range(len(lbd_new)):
    #             lbd = lbd_new[tr_index]
    #             x = X[simu_index][tr_index]
    #             f*= lbd*np.exp(-lbd*x)
    #         f_array_list[lbd_index][simu_index] = f
    
    # Create a mask for simulations where fail_time < obs_time
    #mask = fail_times[n_run_previous:n_run] < obs_time

    # Filter X and fail_times arrays based on the mask
    #X_masked = X[n_run_previous:n_run][mask]

    # Vectorized computation for `f`
    #lbd_exp_values = np.exp(-lbd_new[:, np.newaxis] * X_masked.T)
    lbd_exp_values = np.exp(-lbd_new[:, np.newaxis] * X[n_run_previous:n_run].T)
    f_array = np.prod(lbd_new[:, np.newaxis] * lbd_exp_values, axis=0)*(X[n_run_previous:n_run,0]*X[n_run_previous:n_run,1]>0)

    # Assign computed values back to f_array_list at the corresponding index
    #f_array_list[lbd_index][n_run_previous:n_run][mask] = f_array
    f_array_list[lbd_index][n_run_previous:n_run]= f_array
    # f_array multiplied by indicator function of failure zone
    f_array_ind = f_array_list[lbd_index].copy()
    f_array_ind[fail_times>=obs_time]=0
    #if n_run-n_run_previous != :
    #    print(f"problem n_run-n_run_previous = {n_run-n_run_previous}")
    #    raise Exception("n_run-n_run_previous not adequate")
    if len(f_array_list[lbd_index]) == 0:
        raise Exception("f_array empty 2")
    #print(f"\n f_array_list: {f_array_list}")
    p_f,error,var,same_sample_bool = best_combination_estimation_ESS(true_pf,var_list,p_f_previous,var_previous,error_previous,f_array_ind,f_array_list[lbd_index],g_array_list,g_list_duplicates,n_run-n_run_previous)
    if len(f_array_list[lbd_index]) == 0:
        raise Exception("f_array empty 3")
    return p_f,error,var,same_sample_bool

def compute_pf_cv_baseline(lbd_new,X,fail_times,obs_time,g_array,f_array_list,lbd_index):
    """
    Compute failure probability and coefficient of variation for an IS simulations until obs_time
    with a different lambda that the one used for sampling
    with a list of sampling density for multiple IS
    """
    n_run = len(X)
    n_run_previous = len(f_array_list[lbd_index])

    #update f
    f_array_list[lbd_index] = np.pad(f_array_list[lbd_index],(0,n_run-n_run_previous))
    if len(f_array_list[lbd_index]) == 0:
        raise Exception("f_array empty 1")

    # Vectorized computation for `f`
    lbd_exp_values = np.exp(-lbd_new[:, np.newaxis] * X[n_run_previous:n_run].T)
    f_array = np.prod(lbd_new[:, np.newaxis] * lbd_exp_values, axis=0)*(X[n_run_previous:n_run,0]*X[n_run_previous:n_run,1]>0)
    # Assign computed values back to f_array_list at the corresponding index
    f_array_list[lbd_index][n_run_previous:n_run]= f_array

    f_g = f_array_list[lbd_index]/g_array

    # f_g multiplied by indicator function of failure zone
    f_g_ind = f_g.copy()
    f_g_ind[fail_times>=obs_time]=0

    p_f = np.mean(f_g_ind)
    var = np.var(f_g_ind)/len(f_g_ind)
    ESS = np.sum(f_g)**2/np.sum(f_g**2)
    return p_f,var,ESS

def compute_pf_cv_mult_alt2(true_pf,var_list,p_f_previous,var_previous,ess_previous,error_previous,lbd_new,g_list_duplicates,X,fail_times,obs_time,g_array_list,f_array_list,lbd_index,update_list):
    """
    Compute failure probability and coefficient of variation for an IS simulations until obs_time
    with a different lambda that the one used for sampling
    with a list of sampling density for multiple IS
    """
    n_run = len(X)
    n_run_previous = len(f_array_list[lbd_index])

    #update f
    f_array_list[lbd_index] = np.pad(f_array_list[lbd_index],(0,n_run-n_run_previous))
    if len(f_array_list[lbd_index]) == 0:
        raise Exception("f_array empty 1")

    
    # for simu_index in range(n_run_previous,n_run):
    #     # if failure before obs_time
    #     if fail_times[simu_index]<obs_time:
    #         # compute f
    #         f = 1
    #         for tr_index in range(len(lbd_new)):
    #             lbd = lbd_new[tr_index]
    #             x = X[simu_index][tr_index]
    #             f*= lbd*np.exp(-lbd*x)
    #         f_array_list[lbd_index][simu_index] = f
    
    # Create a mask for simulations where fail_time < obs_time
    #mask = fail_times[n_run_previous:n_run] < obs_time

    # Filter X and fail_times arrays based on the mask
    #X_masked = X[n_run_previous:n_run][mask]

    # Vectorized computation for `f`
    #lbd_exp_values = np.exp(-lbd_new[:, np.newaxis] * X_masked.T)
    lbd_exp_values = np.exp(-lbd_new[:, np.newaxis] * X[n_run_previous:n_run].T)
    f_array = np.prod(lbd_new[:, np.newaxis] * lbd_exp_values, axis=0)*(X[n_run_previous:n_run,0]*X[n_run_previous:n_run,1]>0)

    # Assign computed values back to f_array_list at the corresponding index
    #f_array_list[lbd_index][n_run_previous:n_run][mask] = f_array
    f_array_list[lbd_index][n_run_previous:n_run]= f_array
    # f_array multiplied by indicator function of failure zone
    f_array_ind = f_array_list[lbd_index].copy()
    f_array_ind[fail_times>=obs_time]=0
    #if n_run-n_run_previous != :
    #    print(f"problem n_run-n_run_previous = {n_run-n_run_previous}")
    #    raise Exception("n_run-n_run_previous not adequate")
    if len(f_array_list[lbd_index]) == 0:
        raise Exception("f_array empty 2")
    #print(f"\n f_array_list: {f_array_list}")
    # recheck p_f only if lbd_index in list of index to check
    recheck = lbd_index in update_list
    p_f,error,var,ess_estim,same_sample_bool = best_combination_estimation(true_pf,var_list,p_f_previous,var_previous,error_previous,ess_previous,f_array_ind,f_array_list[lbd_index],g_array_list,g_list_duplicates,n_run-n_run_previous,recheck)

    if len(f_array_list[lbd_index]) == 0:
        raise Exception("f_array empty 3")
    return p_f,error,var,ess_estim,same_sample_bool

def compute_pf_cv_new(true_pf,var_list,p_f_previous,var_previous,ess_previous,error_previous,lbd_new,g_list_duplicates,X,fail_times,obs_time,g_array_list,f_array_list,lbd_index,N):
    """
    Compute failure probability and coefficient of variation for an IS simulations until obs_time
    with a different lambda that the one used for sampling
    with a list of sampling density for multiple IS
    """
    n_run = len(X)
    n_run_previous = len(f_array_list[lbd_index])

    #update f
    f_array_list[lbd_index] = np.pad(f_array_list[lbd_index],(0,n_run-n_run_previous))
    if len(f_array_list[lbd_index]) == 0:
        raise Exception("f_array empty 1")

    lbd_exp_values = np.exp(-lbd_new[:, np.newaxis] * X[n_run_previous:n_run].T)
    f_array = np.prod(lbd_new[:, np.newaxis] * lbd_exp_values, axis=0)*(X[n_run_previous:n_run,0]*X[n_run_previous:n_run,1]>0)

    f_array_list[lbd_index][n_run_previous:n_run]= f_array
    # f_array multiplied by indicator function of failure zone
    f_array_ind = f_array_list[lbd_index].copy()
    f_array_ind[fail_times>=obs_time]=0

    p_f,error,var,ess_estim,same_sample_bool = best_combination_estimation_last(true_pf,var_list,p_f_previous,var_previous,error_previous,ess_previous,f_array_ind,f_array_list[lbd_index],g_array_list,g_list_duplicates,N)

    return p_f,error,var,ess_estim,same_sample_bool

def pick_freeze(p_f,M):
    """
    pick freeze estimator for full sample of size 2M (with variance)
    """
    #return ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
    p_f_M = p_f[:M]
    p_f_M_shift = p_f[M:2*M]

    numerator = (np.mean(p_f_M * p_f_M_shift) - np.mean(p_f_M) * np.mean(p_f_M_shift))
    denominator = (np.mean(p_f_M**2) - np.mean(p_f_M)**2)

    result = numerator / denominator
    return result,denominator


def process_index_total(index, p_f_sample_1, p_f_sample_2, n, n_estim):
    if index % 10 == 0:
        print(f"Processing index: {index}")
    # Copy sample arrays to avoid modifying shared memory
    p_f_sample_pick_freeze = np.copy(p_f_sample_2)
    selected_index = [i for i in range(n) if i!=index]
    p_f_sample_pick_freeze[:, selected_index] = p_f_sample_1[:, selected_index]

    # Concatenate X
    full_sample = np.concatenate((p_f_sample_1, p_f_sample_pick_freeze))

    # Compute Y
    full_sample_pick_freeze = np.array([pick_freeze(np.array(p_f_sample), n//2) for p_f_sample in list(full_sample)[0]])

    # Sobol estimation for the given index
    sobol_estimation = 1-pick_freeze(full_sample_pick_freeze, n_estim)[0]
    
    return (index, sobol_estimation)

def process_index_main(index, p_f_sample_1, p_f_sample_2, n, n_estim, display = False):
    """
    Estimate Sobol of Sobol corresponding to input probability of index index
    """
    if index % 10 == 0 and display:
        print(f"Processing index: {index}")
    # Copy sample arrays to avoid modifying shared memory
    p_f_sample_pick_freeze = np.copy(p_f_sample_2)
    selected_index = index
    p_f_sample_pick_freeze[:, selected_index] = p_f_sample_1[:, selected_index]

    # Concatenate X
    full_sample = np.concatenate((p_f_sample_1, p_f_sample_pick_freeze))

    # Compute Y
    full_sample_pick_freeze = np.array([pick_freeze(np.array(p_f_sample), n//2)[0] for p_f_sample in list(full_sample)])

    # Sobol estimation for the given index
    sobol_estimation,var_estimation = pick_freeze(full_sample_pick_freeze, n_estim)
    if sobol_estimation>0.01 and display:
        print(f"Sobol estimation for index {index}: {sobol_estimation}")
    return (index, sobol_estimation,var_estimation)

def select_index_to_improve_old(p_f,var,n_estim):
    """
    Estimate Total order Sobols of pick freeze estimator (with pick-freeze)
    """
    n = len(p_f)
    p_f_sample_1 = np.random.default_rng().normal(p_f, np.sqrt(var), size=(n_estim, n))
    p_f_sample_2 = np.random.default_rng().normal(p_f, np.sqrt(var), size=(n_estim, n))#np.array([[np.random.default_rng().normal(p_f[i],np.sqrt(var[i])) for i in range(n)] for _ in range(n_estim)])
    sobol_estimations = np.zeros(n)

    partial_process_index = ft.partial(process_index_main, 
                                    p_f_sample_1=p_f_sample_1, 
                                    p_f_sample_2=p_f_sample_2, 
                                    n=n, 
                                    n_estim=n_estim,
                                    display=True)

    # estimate all Sobol of order 1
    for index in range(n):
        sobol_estimations[index] = partial_process_index(index)[1]
    # with mp.Pool(10) as pool:
    #     for result in pool.imap_unordered(partial_process_index, range(n)):
    #         index, sobol_estimation = result
    #         sobol_estimations[index] = sobol_estimation


    return sobol_estimations

def pick_freeze_alt(p_f):
    """
    pick freeze estimator for full sample of size 2M (with variance)
    """
    #return ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
    M = len(p_f)//2

    result = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
    return [result]



def select_index_to_improve(p_f,var,n_estim):
    """
    Estimate Total order Sobols of pick freeze estimator (with rank estimator)
    """
    n = len(p_f)
    ot_Sobol = ot.PythonFunction(n,1,pick_freeze_alt)

    p_f_dist = ot.Normal(p_f,np.sqrt(var))
    p_f_sample = p_f_dist.getSample(n_estim)
    output_sample = ot_Sobol(p_f_sample)

    # estimate all Sobol of order 1
    mySobol = otexp.RankSobolSensitivityAlgorithm(p_f_sample, output_sample)
    indices = mySobol.getFirstOrderIndices()

    return np.array(indices)

def compute_ksi(p_f,var,n_estim,index,n_dist = 100,reduction_factor_list=[0.4]):
    """
    Estimate ksi for factor prioritization
    """
    n = len(p_f)
    p_f_sample_1 = np.random.default_rng().normal(p_f, np.sqrt(var), size=(n_estim, n))
    p_f_sample_2 = np.random.default_rng().normal(p_f, np.sqrt(var), size=(n_estim, n))#np.array([[np.random.default_rng().normal(p_f[i],np.sqrt(var[i])) for i in range(n)] for _ in range(n_estim)])
    # compute base first sobol index


    
    _,base_sobol_estimation,base_var_estimation = process_index_main(index,p_f_sample_1,p_f_sample_2,n,n_estim)
    print(f"base Sobol: {base_sobol_estimation}")
    print(f"initial variance of X_index : {var[index]}")
    ksi_array = np.zeros(len(reduction_factor_list))
    # estimate ksi for each reduction factor
    for reduction_factor_index in range(len(reduction_factor_list)):
        reduction_factor = reduction_factor_list[reduction_factor_index]
        print(f"\ncomputing ksi for reduction factor {reduction_factor}")
        # estimate variance and Sobol for n_dist different input distribution with reduced variance
        sobol_new_dist = np.zeros(n_dist)
        var_new_dist = np.zeros(n_dist)
        for dist_index in range(n_dist):
            # new mean for distribution of p_f[index]
            new_mean = np.random.default_rng().uniform(p_f[index]-(1-np.sqrt(reduction_factor))*np.sqrt(var[index]),p_f[index]+(1-np.sqrt(reduction_factor))*np.sqrt(var[index]))
            # new samples for p_f[index] with new distribution
            new_p_sample_1 = np.random.default_rng().normal(new_mean, np.sqrt(reduction_factor*var[index]), size=n_estim)
            new_p_sample_2 = np.random.default_rng().normal(new_mean, np.sqrt(reduction_factor*var[index]), size=n_estim)

            new_sample_1 = p_f_sample_1.copy()
            new_sample_1[:,index] = new_p_sample_1

            new_sample_2 = p_f_sample_2.copy()
            new_sample_2[:,index] = new_p_sample_2

            _,sobol_new_dist[dist_index],var_new_dist[dist_index] = process_index_main(index,new_sample_1,new_sample_2,n,n_estim)

        # estimate E[var(Y')S_i'|lambda_i] (i=index)
        E_lambda_i = np.mean(var_new_dist*sobol_new_dist)
        #print(f"variance of new X_index: {np.var(new_p_sample_1)}")
        print(f"mean estimated Sobol: {np.mean(sobol_new_dist)}")

        # estimation of the sensitivity index associated to i(=index)
        ksi_array[reduction_factor_index] = (base_var_estimation*base_sobol_estimation-E_lambda_i)/base_var_estimation
        print(f"ksi = {ksi_array[reduction_factor_index]}")

    return ksi_array


def var_estimation_bootstrap(p_f,M,bootstrap_size = 1000,n_bootstrap = 1000):
    """
    Estimate variance (and mean) of pick-freeze estimator with bootstrap 
    Bootstrap done on M first p, and respective associated p in [M:2M] picked
    """
    # default values
    if bootstrap_size is None:
        bootstrap_size = M//2
    if n_bootstrap is None:
        n_bootstrap = M//2
    # indices to pick
    indices_values = [i for i in range(M)]
    new_p_samples = np.zeros(n_bootstrap)
    for sample_index in range(n_bootstrap):
        new_indices = np.random.choice(indices_values,size=bootstrap_size)
        p_f_new = np.concatenate((p_f[new_indices],p_f[new_indices+M]))
        new_p_samples[sample_index] = pick_freeze_alt(p_f)[0]
    var = np.var(new_p_samples)
    mean = np.mean(new_p_samples)
    return var,mean


def compute_Sobol_confidence_interval(S,p_f,M):
    """
    Estimate 95% confidence interval for pick-freeze Sobol estimator
    """
    Y = p_f[:M]
    Y_1 = p_f[M:2*M]
    Y_var = np.var(Y**2)
    sigma = (np.var(Y*Y_1)-2*S*np.cov([Y*Y_1,Y**2])[0][1]+S**2*Y_var)/Y_var
    lower_bound = S-1.96*sigma/np.sqrt(M)
    higher_bound = S+1.96*sigma/np.sqrt(M)
    return sigma,lower_bound,higher_bound

def Sobol_mIS_adaptive(lbd_g,lbd_indices_v,parameter_list,T,N,M,alpha,eps):
    """
    Sobol indices of lambda of indices lbd_indices_v, with lamba drawn according to draw_lbd
    Initial sampling density lbd_g obtained for a lbd_0
    Computation with multiple importance sampling
    
    """

    # default AIS-CE start point
    lbd_start = [[0,0],np.array([[1,0],[0,1]])]
    
    # initial lbd for the sampling density
    #_,_,lbd_g = AIS_CE(lbd_0,parameter_list,obs_name,T,max_time,N,alpha,[i for i in range(len(parameter_list))])
    #print("initial IS parameters:")
    #for x_index in range(len(lbd_g)):
    #    print(f"{parameter_list[x_index][1]} ({x_index}): {lbd_g[x_index]}")
    # def create_normal_drawer(lbd_t):
    #     return np.random.default_rng().normal(loc=lbd_t[0], scale=lbd_t[1])
    # draw_g = [create_normal_drawer(lbd_i) for lbd_i in lbd_g]
    #draw_g = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd_g]
    draw_g = (lambda lbd_t = lbd_g : np.random.default_rng().multivariate_normal(mean = lbd_t[0], cov = lbd_t[1]))

    # def create_gaussian_function(lbd_t):
    #     def gaussian(x):
    #         return 1 / (np.sqrt(2 * np.pi) * lbd_t[1]) * np.exp(-0.5 * ((x - lbd_t[0]) / lbd_t[1]) ** 2)
    #     return gaussian

    # g = [create_gaussian_function(lbd_i) for lbd_i in lbd_g]
    #g = [(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g]
    g = lambda x, lbd_t=lbd_g: 1/(2*np.pi*np.linalg.det(lbd_t[1]))*np.exp(-0.5*(np.array([x-lbd_t[0]])@np.linalg.inv(lbd_t[1])@np.array([x-lbd_t[0]]).T)[0,0])

    # draw samples according to the density
    X,_,fail_times = sample_IS(draw_g,parameter_list,N)
    
    # for test
    #X_copy = X.copy()
    #fail_times_copy = fail_times.copy()
    # draw lbd samples for pick freeze
    #draw_lbd = [(lambda lbd_t=lbd_i: np.random.default_rng().uniform(lbd_t-0.1*lbd_t,lbd_t+0.1*lbd_t)) for lbd_i in [0.1,0.1]]

    lbd_samples = np.array([[np.random.default_rng().uniform(0.1,0.3) for tr_index in range(len(parameter_list))] for _ in range(M)])
    lbd_samples_v = np.array([[np.random.default_rng().uniform(0.1,0.3) for tr_index in range(len(parameter_list))] for _ in range(M)])
    #lbd_samples = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
    #lbd_samples_v = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
    for index in lbd_indices_v:
        lbd_samples_v[:,index] = lbd_samples[:,index].copy()
    full_lbd_samples = np.concatenate((lbd_samples,lbd_samples_v))
    # list of sampling functions obtained with AIS-CE
    g_list = [g]
    # list that indicates if the g of the corresponding index in g_list is a duplicate of the previous g in the list, and how many duplicates of g there is
    g_list_duplicates = [0]
    
    n_IS = 1
    cv_max = 1
    g_array_list = []
    f_array_list = [np.zeros(0) for _ in full_lbd_samples]
    #lbd_start = lbd_g.copy()
    # estimation with lambda samples and true probabilities
    p_true = np.array([true_failure_proba(*lbd_sample) for lbd_sample in full_lbd_samples])
    print(f"True probas : {p_true}")
    S_true = ((1/M)*np.sum([p_true[i]*p_true[i+M]  for i in range(M)])-(1/M)*np.sum([p_true[i] for i in range(M)])* (1/M)*np.sum([p_true[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_true[i]**2 for i in range(M)])- ((1/M)*np.sum([p_true[i]  for i in range(M)]))**2)
    print(f"Sobol estimation with true probabilities: {S_true}")
    p_f = [1 for _ in full_lbd_samples]
    
    cv = [np.inf for _ in full_lbd_samples]
    # ESS
    #cv = [0 for _ in full_lbd_samples]
    var = [np.inf for _ in full_lbd_samples]
    ess = [1 for _ in full_lbd_samples]
    estim_cv = [np.inf for _ in full_lbd_samples]
    #useful_X_index = [np.zeros(0) fo _ in full_lbd_samples]
    var_lists = [[] for _ in full_lbd_samples]
    # previous arg for cv_max for comparison, to avoid recomputation of AIS-CE in case of same index for cv_max
    cv_arg_max_previous = -1
    #var_arg_max_previous = -1

    # index to update
    update_list = [i for i in range(2*M)]
    
    while cv_max>eps:
    #while n_IS<7:
        print(f"\nn_IS = {n_IS}")

        # update g_array_list
        update_g_array_list(lbd_g,g_list,g_array_list,X,fail_times,T,N)
        print("g_array_list updated")
        # compute p_T estimations for all lbd samples
        # if n_IS == 1:
        p_f_and_cv = np.array([compute_pf_cv_mult_alt2(p_true[lbd_index],var_lists[lbd_index],p_f[lbd_index],var[lbd_index],ess[lbd_index],cv[lbd_index],full_lbd_samples[lbd_index],g_list_duplicates,X,fail_times,T,g_array_list,f_array_list,lbd_index,update_list) for lbd_index in range(len(full_lbd_samples))])
        if n_IS>=2:
            print(f"new estimated proba for selected index for IS: {p_f_and_cv[cv_arg_max,0]}")
            print(f"corresponding new estimation variance: {p_f_and_cv[cv_arg_max,2]}")
        # else:
        #     p_f_and_cv[cv_arg_max] = compute_pf_cv_new(p_true[cv_arg_max],var_lists[cv_arg_max],p_f[cv_arg_max],var[cv_arg_max],ess[cv_arg_max],cv[cv_arg_max],full_lbd_samples[cv_arg_max],g_list_duplicates,X,fail_times,T,g_array_list,f_array_list,cv_arg_max,N)
        #     print(f"new estimated proba: {p_f_and_cv[cv_arg_max,0]}")
        #     print(f"new estimation variance: {p_f_and_cv[cv_arg_max,2]}")
        #p_f_and_cv = np.array([compute_pf_cv_mult_alt(p_true[lbd_index],var_lists[lbd_index],p_f[lbd_index],var[lbd_index],cv[lbd_index],full_lbd_samples[lbd_index],g_list_duplicates,X,fail_times,T,g_array_list,f_array_list,lbd_index) for lbd_index in range(len(full_lbd_samples))])

        #print(f"p_f_and_cv: {p_f_and_cv}")
        #p_f_and_cv = np.array([[0.5,0.5,False],[0.5,0.5,False]])
        #gc.collect()


        
        p_f = p_f_and_cv[:,0]
        #p_f2 = p_f_and_cv2[:,0]
        print(f"estimated failure proba: {p_f}")
        #print(f"estimated failure proba 2: {p_f2}")
        #print(f"10 first estimated failure proba: {p_f[:10]}")
        # relative error for probas estimations
        error = np.abs(((np.array(p_f)/p_true)-1))
        print(f"relative errors for probability estimations : {np.sort(error)}")
        print(f"mean relative error : {np.mean(error)}")
        cv = p_f_and_cv[:,1]
        var = p_f_and_cv[:,2]
        ess = p_f_and_cv[:,3]
        #var2 = p_f_and_cv2[:,2]
        estim_cv = np.sqrt(np.array(var))/np.array(p_f)
        #estim_cv2 = np.sqrt(np.array(var2))/np.array(p_f2)
        #print(f"estimed cv : {estim_cv}")
        print(f"mean cv : {np.mean(estim_cv)}")
        print(f"max cv : {np.max(estim_cv)}")
        print(f"mean ess : {np.mean(ess)}")
        print(f"min ess : {np.min(ess)}")
        #print(f"estimed cv2 : {estim_cv2}")
        S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
        print(f"current Sobol index estimation: {S}")

        # estimation of variance of pick_freeze estimator with bootstrap
        # var_bootstrap,mean_bootstrap = var_estimation_bootstrap(p_f,M)
        # print(f"variance of estimator estimation with bootstrap:{var_bootstrap}")
        # print(f"cv of estimator estimation with bootstrap:{np.sqrt(var_bootstrap)/mean_bootstrap}\n")

        # compute confidence interval
        sigma,lower_bound,higher_bound = compute_Sobol_confidence_interval(S,p_f,M)
        print(f"estimated std: {sigma/np.sqrt(N)}")
        print(f"estimated 95% confidence interval: [{lower_bound},{higher_bound}]\n")

        # check if kept sample didn't change
        same_sample = p_f_and_cv[:,4]
        if np.prod(same_sample):
            print("\n no additionnal X sample used for the new estimation \n")
        # check first parameters with highest cv (or ESS)
        # Total Sobol criterion
        cv = select_index_to_improve(p_f,var,2000)
        print(f"sum of first order Sobols: {np.sum(cv[cv>0])}")
        print(f"number of positive Sobols: {np.sum(cv>0)}")
        print(f"number of Sobols superior to 10%: {np.sum(cv>0.1)}")
        print(f"paramaters corresponding to positive Sobol: {full_lbd_samples[cv>0]}")
        # update only parameter with Sobol superior to 1%
        update_list = np.where(cv>0.01)[0]
        print(f"number of potential probability updates : {len(update_list)}")
        cv_argsort = np.argsort(cv)
        cv_arg_max = np.argmax(cv)
        print(f"parameter yielding max error : {full_lbd_samples[cv_arg_max]}")
        print(f"max error : {cv[cv_arg_max]}")
        print(f"corresponding true proba: {p_true[cv_arg_max]}")
        print(f"corresponding estimated proba: {p_f[cv_arg_max]}")
        print(f"corresponding estimated variance: {var[cv_arg_max]}")
        print(f"corresponding estimated cv: {np.sqrt(var[cv_arg_max])/p_f[cv_arg_max]}")
        #compute_ksi(p_f,var,1000,cv_arg_max,1000,[1.,0.9,0.5,0.1,1e-9,0.])

        # ESS min
        #cv_arg_min = np.argmin(cv)
        cv_max = cv[cv_arg_max]
        lbd_max = full_lbd_samples[cv_arg_max]
        # cv_max = cv[cv_arg_min]
        # lbd_max = full_lbd_samples[cv_arg_min]

        # check first parameters with highest var
        # var_argsort = np.argsort(var)
        # var_arg_max = np.argmax(var)
        # var_max = var[var_arg_max]
        # lbd_max = full_lbd_samples[var_arg_max]

        #print(f"error max: {cv_max}")
        #print(f"corresponding true proba: {p_true[cv_arg_max]}")
        #print(f"corresponding estimated proba: {p_f[cv_arg_max]}")

        # print(f"ESS min: {cv_max}")
        # print(f"corresponding true proba: {p_true[cv_arg_min]}")
        # print(f"corresponding estimated proba: {p_f[cv_arg_min]}")
        # print(f"min ESS: {cv_max}")
        # print(f"var max: {var_max}")
        # print(f"mean cv: {np.mean(cv)}")

        #print(f"2 highest error: {[cv[cv_argsort[-i]] for i in range(1,3)]}")
        #print(f"corresponding parameters: {[full_lbd_samples[cv_argsort[-i]] for i in range(1,3)]}")
        # print(f"2 lowest ESS: {[cv[cv_argsort[i-1]] for i in range(1,3)]}")
        # print(f"corresponding parameters: {[full_lbd_samples[cv_argsort[i-1]] for i in range(1,3)]}")

        # print(f"2 highest var: {[var[var_argsort[-i]] for i in range(1,3)]}")
        # print(f"corresponding parameters: {[full_lbd_samples[var_argsort[-i]] for i in range(1,3)]}")
        cv = p_f_and_cv[:,1]
        cv_max= np.max(cv)
        if cv_max > eps:
        # if ESS inferior to threshold
        # if cv_max < eps:
        #if var_max > eps:
        # if cv_arg_max different from previous cv_arg_max, do AIS-CE, else reuse same
            if cv_arg_max != cv_arg_max_previous:
            # if cv_arg_min != cv_arg_max_previous:
            #if var_arg_max != var_arg_max_previous:

                # start with previous point
                
                _,_,lbd_g_new,_ = AIS_CE(lbd_start,lbd_max,parameter_list,T,0,10000,alpha,[i for i in range(len(parameter_list))])
                #lbd_g_new = lbd_g
                # print(f"lbd_g : {lbd_g}")
                # print(f"lbd_g new : {lbd_g_new}")
                #_,_,lbd_g_new,_ = AIS_CE(lbd_start,lbd_max,parameter_list,T,0,N,alpha,[i for i in range(len(parameter_list))])
                #lbd_g_new = lbd_start
                print("AIS_CE completed")
                #lbd_g_new = lbd_g.copy()
                #print("new IS parameters")
                #for x_index in range(len(lbd_g_new)):
                #    print(f"{parameter_list[x_index][1]} ({x_index}): {lbd_g_new[x_index]}")
                #draw_g_new = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd_g_new]
                draw_g_new = (lambda lbd_t = lbd_g_new : np.random.default_rng().multivariate_normal(mean = lbd_t[0], cov = lbd_t[1]))
                g_list+=[lambda x, lbd_t=lbd_g_new: 1/(2*np.pi*np.linalg.det(lbd_t[1]))*np.exp(-0.5*(np.array([x-lbd_t[0]])@np.linalg.inv(lbd_t[1])@np.array([x-lbd_t[0]]).T)[0,0])]#[[(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g_new]]
                g_list_duplicates+=[0]
                # draw samples according to the density
                X_new,_,fail_times_new = sample_IS(draw_g_new,parameter_list,N)
                print(f"proportion of new samples in failure zone: {np.mean(fail_times_new<T)}")
                #X_new = X_copy.copy()
                #fail_times_new = fail_times_copy.copy()

        # if cv_arg_max == cv_arg_max_previous:
        #     print("lbd index yielding cv_max is the same as previously")
            else:
                
                print("lbd index yielding cv_max is the same as previously, reuse of previous AIS-CE results")
                # print("lbd index yielding cv_max is the same as previously, new AIS-CE for same lbd")
                # _,_,lbd_g_new,_ = AIS_CE(lbd_start,lbd_max,parameter_list,T,0,10000,alpha,[i for i in range(len(parameter_list))])
                # draw_g_new = (lambda lbd_t = lbd_g_new : np.random.default_rng().multivariate_normal(mean = lbd_t[0], cov = lbd_t[1]))
                X_new,_,fail_times_new = sample_IS(draw_g_new,parameter_list,N)
                print(f"proportion of new samples in failure zone: {np.mean(fail_times_new<T)}")
                g_list+=[lambda x, lbd_t=lbd_g_new: 1/(2*np.pi*np.linalg.det(lbd_t[1]))*np.exp(-0.5*(np.array([x-lbd_t[0]])@np.linalg.inv(lbd_t[1])@np.array([x-lbd_t[0]]).T)[0,0])]#[[(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g_new]]
                g_list_duplicates+=[g_list_duplicates[-1]+1]
                # update previous g_list_duplicates indices
                for duplicate_index in range(g_list_duplicates[-1]):
                    g_list_duplicates[len(g_list_duplicates)-2-duplicate_index] = g_list_duplicates[-1]
            print(f"g_list_duplicates : {g_list_duplicates}")
            # update samples and g (multiple IS)
            X = np.concatenate((X,X_new))
            fail_times =np.concatenate((fail_times,fail_times_new))
            #X = np.concatenate((X,X))
            #fail_times =np.concatenate((fail_times,fail_times))

            #lbd_start = lbd_g_new.copy()
            cv_arg_max_previous = cv_arg_max#cv_arg_min #
            #var_arg_max_previous = var_arg_max
            n_IS+=1
            """            
            print(f'\n stats check\n')
            print(len(f_array_list[0]))
            print(len(g_array_list))
            print(len(g_array_list[0]))
            print(len(cv_lists))
            print(len(cv_lists[0]))
            print(f'\n')
            """
        #gc.collect()
    print(f"final cv: {cv_max}")

    S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)

    return S,g,X

def Sobol_mIS_ad_comparison(lbd_g,lbd_indices_v,parameter_list,T,N,M,alpha,eps):
    """
    Sobol indices of lambda of indices lbd_indices_v, with lamba drawn according to draw_lbd
    Initial sampling density lbd_g obtained for a lbd_0
    Computation with multiple importance sampling
    
    """

    # default AIS-CE start point
    lbd_start = [[0,0],np.array([[1,0],[0,1]])]
    
    # initial lbd for the sampling density
    #_,_,lbd_g = AIS_CE(lbd_0,parameter_list,obs_name,T,max_time,N,alpha,[i for i in range(len(parameter_list))])
    #print("initial IS parameters:")
    #for x_index in range(len(lbd_g)):
    #    print(f"{parameter_list[x_index][1]} ({x_index}): {lbd_g[x_index]}")
    # def create_normal_drawer(lbd_t):
    #     return np.random.default_rng().normal(loc=lbd_t[0], scale=lbd_t[1])
    # draw_g = [create_normal_drawer(lbd_i) for lbd_i in lbd_g]
    #draw_g = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd_g]
    draw_g = (lambda lbd_t = lbd_g : np.random.default_rng().multivariate_normal(mean = lbd_t[0], cov = lbd_t[1]))

    # def create_gaussian_function(lbd_t):
    #     def gaussian(x):
    #         return 1 / (np.sqrt(2 * np.pi) * lbd_t[1]) * np.exp(-0.5 * ((x - lbd_t[0]) / lbd_t[1]) ** 2)
    #     return gaussian

    # g = [create_gaussian_function(lbd_i) for lbd_i in lbd_g]
    #g = [(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g]
    g = lambda x, lbd_t=lbd_g: 1/(2*np.pi*np.linalg.det(lbd_t[1]))*np.exp(-0.5*(np.array([x-lbd_t[0]])@np.linalg.inv(lbd_t[1])@np.array([x-lbd_t[0]]).T)[0,0])

    # draw samples according to the density
    X,_,fail_times = sample_IS(draw_g,parameter_list,N)
    X_baseline = X.copy()
    fail_times_baseline = fail_times.copy()
    # for test
    #X_copy = X.copy()
    #fail_times_copy = fail_times.copy()
    # draw lbd samples for pick freeze
    #draw_lbd = [(lambda lbd_t=lbd_i: np.random.default_rng().uniform(lbd_t-0.1*lbd_t,lbd_t+0.1*lbd_t)) for lbd_i in [0.1,0.1]]

    lbd_samples = np.array([[np.random.default_rng().uniform(0.01,0.4) for tr_index in range(len(parameter_list))] for _ in range(M)])
    lbd_samples_v = np.array([[np.random.default_rng().uniform(0.01,0.4) for tr_index in range(len(parameter_list))] for _ in range(M)])
    #lbd_samples = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
    #lbd_samples_v = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
    for index in lbd_indices_v:
        lbd_samples_v[:,index] = lbd_samples[:,index].copy()
    full_lbd_samples = np.concatenate((lbd_samples,lbd_samples_v))
    # list of sampling functions obtained with AIS-CE
    g_list = [g]
    # list that indicates if the g of the corresponding index in g_list is a duplicate of the previous g in the list, and how many duplicates of g there is
    g_list_duplicates = [0]
    
    n_IS = 1
    cv_max = 1
    g_array_list = []
    g_array_baseline = np.zeros(0)
    f_array_list = [np.zeros(0) for _ in full_lbd_samples]
    f_array_list_baseline = [np.zeros(0) for _ in full_lbd_samples]
    #lbd_start = lbd_g.copy()
    # estimation with lambda samples and true probabilities
    p_true = np.array([true_failure_proba(*lbd_sample) for lbd_sample in full_lbd_samples])
    print(f"True probas : {p_true}")
    S_true = ((1/M)*np.sum([p_true[i]*p_true[i+M]  for i in range(M)])-(1/M)*np.sum([p_true[i] for i in range(M)])* (1/M)*np.sum([p_true[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_true[i]**2 for i in range(M)])- ((1/M)*np.sum([p_true[i]  for i in range(M)]))**2)
    print(f"Sobol estimation with true probabilities: {S_true}")
    p_f = [1 for _ in full_lbd_samples]
    p_f_baseline = [1 for _ in full_lbd_samples]
    
    #cv = [np.inf for _ in full_lbd_samples]
    # ESS
    cv = [0 for _ in full_lbd_samples]
    var = [np.inf for _ in full_lbd_samples]
    #ess = [1 for _ in full_lbd_samples]
    estim_cv = [np.inf for _ in full_lbd_samples]
    #useful_X_index = [np.zeros(0) fo _ in full_lbd_samples]
    var_lists = [[] for _ in full_lbd_samples]
    # previous arg for cv_max for comparison, to avoid recomputation of AIS-CE in case of same index for cv_max
    cv_arg_max_previous = -1
    #var_arg_max_previous = -1
    while cv_max<eps:
    #while n_IS<7:
        print(f"\nn_IS = {n_IS}")

        # update g_array_list
        update_g_array_list(lbd_g,g_list,g_array_list,X,fail_times,T,N)
        g_array_baseline = update_g_array_baseline(g,g_array_baseline,X_baseline)
        print("g_array_list updated")
        # compute p_T estimations for all lbd samples
        #p_f_and_cv = np.array([compute_pf_cv_mult_alt2(p_true[lbd_index],var_lists[lbd_index],p_f[lbd_index],var[lbd_index],ess[lbd_index],cv[lbd_index],full_lbd_samples[lbd_index],g_list_duplicates,X,fail_times,T,g_array_list,f_array_list,lbd_index) for lbd_index in range(len(full_lbd_samples))])
        p_f_and_cv = np.array([compute_pf_cv_mult_alt(p_true[lbd_index],var_lists[lbd_index],p_f[lbd_index],var[lbd_index],cv[lbd_index],full_lbd_samples[lbd_index],g_list_duplicates,X,fail_times,T,g_array_list,f_array_list,lbd_index) for lbd_index in range(len(full_lbd_samples))])
        p_f_and_cv_baseline = np.array([compute_pf_cv_baseline(full_lbd_samples[lbd_index],X_baseline,fail_times_baseline,T,g_array_baseline,f_array_list_baseline,lbd_index) for lbd_index in range(len(full_lbd_samples))])

        #print(f"p_f_and_cv: {p_f_and_cv}")
        #p_f_and_cv = np.array([[0.5,0.5,False],[0.5,0.5,False]])
        #gc.collect()


        
        p_f = p_f_and_cv[:,0]
        p_f_baseline = p_f_and_cv_baseline[:,0]
        #p_f2 = p_f_and_cv2[:,0]

        #print(f"estimated failure proba: {p_f}")

        #print(f"estimated failure proba 2: {p_f2}")
        #print(f"10 first estimated failure proba: {p_f[:10]}")
        # relative error for probas estimations
        print(f"\nADAPTIVE MIS RESULTS")
        error = np.abs(((np.array(p_f)/p_true)-1))
        print(f"relative errors for probability estimations : {np.sort(error)}")
        print(f"mean relative error : {np.mean(error)}")
        cv = p_f_and_cv[:,1]
        var = p_f_and_cv[:,2]
        #ess = p_f_and_cv[:,3]
        #var2 = p_f_and_cv2[:,2]
        estim_cv = np.sqrt(np.array(var))/np.array(p_f)
        #estim_cv2 = np.sqrt(np.array(var2))/np.array(p_f2)
        #print(f"estimed cv : {estim_cv}")
        print(f"mean cv : {np.mean(estim_cv)}")
        print(f"max cv : {np.max(estim_cv)}")
        # print(f"mean ess : {np.mean(ess)}")
        # print(f"min ess : {np.min(ess)}")
        #print(f"estimed cv2 : {estim_cv2}")

        
        S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
        print(f"current Sobol index estimation: {S}")
        
        # check if kept sample didn't change
        same_sample = p_f_and_cv[:,3]
        if np.prod(same_sample):
            print("\n no additionnal X sample used for the new estimation \n")
        # check first parameters with highest cv (or ESS)
        cv_argsort = np.argsort(cv)
        #cv_arg_max = np.argmax(cv)
        # ESS min
        cv_arg_min = np.argmin(cv)
        # cv_max = cv[cv_arg_max]
        # lbd_max = full_lbd_samples[cv_arg_max]
        cv_max = cv[cv_arg_min]
        lbd_max = full_lbd_samples[cv_arg_min]

        # check first parameters with highest var
        # var_argsort = np.argsort(var)
        # var_arg_max = np.argmax(var)
        # var_max = var[var_arg_max]
        # lbd_max = full_lbd_samples[var_arg_max]
        # print(f"error max: {cv_max}")
        # print(f"corresponding true proba: {p_true[cv_arg_max]}")
        # print(f"corresponding estimated proba: {p_f[cv_arg_max]}")
        print(f"ESS min: {cv_max}")
        print(f"corresponding true proba: {p_true[cv_arg_min]}")
        print(f"corresponding estimated proba: {p_f[cv_arg_min]}")
        # print(f"min ESS: {cv_max}")
        # print(f"var max: {var_max}")
        # print(f"mean cv: {np.mean(cv)}")

        # print(f"2 highest error: {[cv[cv_argsort[-i]] for i in range(1,3)]}")
        # print(f"corresponding parameters: {[full_lbd_samples[cv_argsort[-i]] for i in range(1,3)]}")
        print(f"2 lowest ESS: {[cv[cv_argsort[i-1]] for i in range(1,3)]}")
        print(f"corresponding parameters: {[full_lbd_samples[cv_argsort[i-1]] for i in range(1,3)]}")
        print("------------------------------------------")

        print("\n BASELINE RESULTS")
        # relative error for probas estimations
        error_baseline = np.abs(((np.array(p_f_baseline)/p_true)-1))
        print(f"relative errors for probability estimations : {np.sort(error_baseline)}")
        print(f"mean relative error : {np.mean(error_baseline)}")
        var_baseline = p_f_and_cv_baseline[:,1]
        ess_baseline = p_f_and_cv_baseline[:,2]

        estim_cv_baseline = np.sqrt(np.array(var_baseline))/np.array(p_f_baseline)

        print(f"mean cv : {np.mean(estim_cv_baseline)}")
        print(f"max cv : {np.max(estim_cv_baseline)}")
        print(f"ESS min: {np.min(ess_baseline)}")


        S = ((1/M)*np.sum([p_f_baseline[i]*p_f_baseline[i+M]  for i in range(M)])-(1/M)*np.sum([p_f_baseline[i] for i in range(M)])* (1/M)*np.sum([p_f_baseline[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f_baseline[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f_baseline[i]  for i in range(M)]))**2)
        print(f"current Sobol index estimation: {S}")
        print("------------------------------------------")
        # print(f"2 highest var: {[var[var_argsort[-i]] for i in range(1,3)]}")
        # print(f"corresponding parameters: {[full_lbd_samples[var_argsort[-i]] for i in range(1,3)]}")
        # if cv_max > eps:
        # if ESS inferior to threshold
        if cv_max < eps:
        #if var_max > eps:
        # if cv_arg_max different from previous cv_arg_max, do AIS-CE, else reuse same
            # if cv_arg_max != cv_arg_max_previous:
            if cv_arg_min != cv_arg_max_previous:
            #if var_arg_max != var_arg_max_previous:

                # start with previous point
                
                _,_,lbd_g_new,n_resample = AIS_CE(lbd_start,lbd_max,parameter_list,T,0,10000,alpha,[i for i in range(len(parameter_list))])
                #lbd_g_new = lbd_g
                # print(f"lbd_g : {lbd_g}")
                # print(f"lbd_g new : {lbd_g_new}")
                #_,_,lbd_g_new,_ = AIS_CE(lbd_start,lbd_max,parameter_list,T,0,N,alpha,[i for i in range(len(parameter_list))])
                #lbd_g_new = lbd_start
                print("AIS_CE completed")
                #lbd_g_new = lbd_g.copy()
                #print("new IS parameters")
                #for x_index in range(len(lbd_g_new)):
                #    print(f"{parameter_list[x_index][1]} ({x_index}): {lbd_g_new[x_index]}")
                #draw_g_new = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd_g_new]
                draw_g_new = (lambda lbd_t = lbd_g_new : np.random.default_rng().multivariate_normal(mean = lbd_t[0], cov = lbd_t[1]))
                g_list+=[lambda x, lbd_t=lbd_g_new: 1/(2*np.pi*np.linalg.det(lbd_t[1]))*np.exp(-0.5*(np.array([x-lbd_t[0]])@np.linalg.inv(lbd_t[1])@np.array([x-lbd_t[0]]).T)[0,0])]#[[(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g_new]]
                g_list_duplicates+=[0]
                # draw samples according to the density
                X_new,_,fail_times_new = sample_IS(draw_g_new,parameter_list,N)
                #print(f"proportion of new samples in failure zone: {np.mean(fail_times_new<T)}")
                #X_new = X_copy.copy()
                #fail_times_new = fail_times_copy.copy()
                # number of calls to black box taken for baseline resample
                n_additionnal_baseline_sample = n_resample*10000+N

        # if cv_arg_max == cv_arg_max_previous:
        #     print("lbd index yielding cv_max is the same as previously")
            else:
                
                print("lbd index yielding cv_max is the same as previously, reuse of previous AIS-CE results")
                # print("lbd index yielding cv_max is the same as previously, new AIS-CE for same lbd")
                # _,_,lbd_g_new,_ = AIS_CE(lbd_start,lbd_max,parameter_list,T,0,10000,alpha,[i for i in range(len(parameter_list))])
                # draw_g_new = (lambda lbd_t = lbd_g_new : np.random.default_rng().multivariate_normal(mean = lbd_t[0], cov = lbd_t[1]))
                X_new,_,fail_times_new = sample_IS(draw_g_new,parameter_list,N)
                #print(f"proportion of new samples in failure zone: {np.mean(fail_times_new<T)}")
                g_list+=[lambda x, lbd_t=lbd_g_new: 1/(2*np.pi*np.linalg.det(lbd_t[1]))*np.exp(-0.5*(np.array([x-lbd_t[0]])@np.linalg.inv(lbd_t[1])@np.array([x-lbd_t[0]]).T)[0,0])]#[[(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g_new]]
                g_list_duplicates+=[g_list_duplicates[-1]+1]
                # update previous g_list_duplicates indices
                for duplicate_index in range(g_list_duplicates[-1]):
                    g_list_duplicates[len(g_list_duplicates)-2-duplicate_index] = g_list_duplicates[-1]
                # number of calls to black box taken for baseline resample
                n_additionnal_baseline_sample = N
            print(f"number of additionnal calls to black box: {n_additionnal_baseline_sample}")
            print(f"g_list_duplicates : {g_list_duplicates}")
            # update samples and g (multiple IS)
            X = np.concatenate((X,X_new))
            fail_times =np.concatenate((fail_times,fail_times_new))
            #X = np.concatenate((X,X))
            #fail_times =np.concatenate((fail_times,fail_times))

            # update baseline
            X_baseline_new,_,fail_times_baseline_new = sample_IS(draw_g,parameter_list,n_additionnal_baseline_sample)
            X_baseline = np.concatenate((X_baseline,X_baseline_new))
            fail_times_baseline =np.concatenate((fail_times_baseline,fail_times_baseline_new))

            #lbd_start = lbd_g_new.copy()
            cv_arg_max_previous = cv_arg_min #cv_arg_max#
            #var_arg_max_previous = var_arg_max
            n_IS+=1
            """            
            print(f'\n stats check\n')
            print(len(f_array_list[0]))
            print(len(g_array_list))
            print(len(g_array_list[0]))
            print(len(cv_lists))
            print(len(cv_lists[0]))
            print(f'\n')
            """
        #gc.collect()
    print(f"final cv: {cv_max}")

    S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)

    return S,g,X

def Sobol_mIS(lbd_g,lbd_indices_v,draw_lbd,parameter_list,T,N,M,alpha,eps):
    """
    Sobol indices of lambda of indices lbd_indices_v, with lamba drawn according to draw_lbd
    Initial sampling density lbd_g obtained for a lbd_0
    Computation with multiple importance sampling
    """
    # initial lbd for the sampling density
    #_,_,lbd_g = AIS_CE(lbd_0,parameter_list,obs_name,T,max_time,N,alpha,[i for i in range(len(parameter_list))])
    #print("initial IS parameters:")
    #for x_index in range(len(lbd_g)):
    #    print(f"{parameter_list[x_index][1]} ({x_index}): {lbd_g[x_index]}")
    draw_g = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd_g]
    g = [(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g]
    # draw samples according to the density
    X,_,fail_times = sample_IS(draw_g,parameter_list,N)
    
    # for test
    #X_copy = X.copy()
    #fail_times_copy = fail_times.copy()
    # draw lbd samples for pick freeze
    lbd_samples = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
    lbd_samples_v = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
    for index in lbd_indices_v:
        lbd_samples_v[:,index] = lbd_samples[:,index].copy()
    full_lbd_samples = np.concatenate((lbd_samples,lbd_samples_v))

    # list of sampling functions obtained with AIS-CE
    g_list = [g]
    
    n_IS = 1
    cv_max = 1
    g_sum_array = np.zeros(0)
    f_array_list = [np.zeros(0) for _ in full_lbd_samples]
    #lbd_start = lbd_g.copy()
    # estimation with lambda samples and true probabilities
    p_f = np.array([true_failure_proba(*lbd_sample) for lbd_sample in full_lbd_samples])
    S_true = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
    print(f"Sobol estimation with true probabilities: {S_true}")
    while cv_max>eps:
    #while n_IS<7:
        print(f"n_IS = {n_IS}")
        
        # compute p_T estimations for all lbd samples
        p_f_and_cv = np.array([compute_pf_cv_mult(full_lbd_samples[lbd_index],g_list,X,fail_times,T,g_sum_array,f_array_list[lbd_index]) for lbd_index in range(len(full_lbd_samples))])
        #gc.collect()
        p_f = p_f_and_cv[:,0]
        #print(f"10 first estimated failure proba: {p_f[:10]}")
        S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
        print(f"current Sobol index estimation: {S}")
        if S!=S:
            for i in range(len(p_f)):
                print(p_f[i])
        cv = p_f_and_cv[:,1]
        # check first parameters with highest cv
        cv_argsort = np.argsort(cv)
        cv_arg_max = np.argmax(cv)
        cv_max = cv[cv_arg_max]
        lbd_max = full_lbd_samples[cv_arg_max]
        print(f"cv max: {cv_max}")
        print(f"5 highest cv: {[cv[cv_argsort[-i]] for i in range(1,6)]}")
        print(f"corresponding parameters: {[full_lbd_samples[cv_argsort[-i]] for i in range(1,6)]}")
        if cv_max > eps:
            # default start point
            lbd_start = np.array([[0.,10.] for _ in range(len(parameter_list))])
            # start with previous point
            
            _,_,lbd_g_new,_ = AIS_CE(lbd_start,lbd_max,parameter_list,T,0,N,alpha,[i for i in range(len(parameter_list))])
            #lbd_g_new = lbd_g.copy()
            #print("new IS parameters")
            #for x_index in range(len(lbd_g_new)):
            #    print(f"{parameter_list[x_index][1]} ({x_index}): {lbd_g_new[x_index]}")
            draw_g_new = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd_g_new]
            g_list+=[[(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g_new]]
            # draw samples according to the density
            X_new,_,fail_times_new = sample_IS(draw_g_new,parameter_list,N)
            #X_new = X_copy.copy()
            #fail_times_new = fail_times_copy.copy()
            # update samples and g (multiple IS)
            X = np.concatenate((X,X_new))
            fail_times =np.concatenate((fail_times,fail_times_new))

            #lbd_start = lbd_g_new.copy()
            n_IS+=1
            gc.collect()
    
    S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)

    return S,g

def compute_all_first_order_Sobol(lbd_g,draw_lbd,parameter_list,T,N,M,alpha,eps):
    S,g,X = Sobol_mIS_adaptive(lbd_g,[0],draw_lbd,parameter_list,T,N,M,alpha,eps)
    for parameter_index in range(1,len(parameter_list)):

        # get lbd samples
        lbd_samples = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
        lbd_samples_v = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
        lbd_samples_v[:,parameter_index] = lbd_samples[:,parameter_index].copy()
        full_lbd_samples = np.concatenate((lbd_samples,lbd_samples_v))

    return

def compare_methods(lbd_g,lbd_indices_v,draw_lbd,parameter_list,T,N,M,alpha,eps):
    """
    Compare max cv of probability estimations with different methods and same budget
    """
    # initial lbd for the sampling density
    draw_g = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd_g]
    g = [(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g]
    # draw samples according to the density
    X,_,fail_times = sample_IS(draw_g,parameter_list,N)
    
    # for test
    #X_copy = X.copy()
    #fail_times_copy = fail_times.copy()
    # draw lbd samples for pick freeze
    lbd_samples = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
    lbd_samples_v = np.array([[draw_lbd[tr_index]() for tr_index in range(len(parameter_list))] for _ in range(M)])
    for index in lbd_indices_v:
        lbd_samples_v[:,index] = lbd_samples[:,index].copy()
    full_lbd_samples = np.concatenate((lbd_samples,lbd_samples_v))

    # list of sampling functions obtained with AIS-CE
    g_list = [g]
    
    n_IS = 1
    cv_max = 1
    g_sum_array = np.zeros(0)
    f_array_list = [np.zeros(0) for _ in full_lbd_samples]
    #lbd_start = lbd_g.copy()
    # estimation with lambda samples and true probabilities
    p_true = p_f = np.array([true_failure_proba(*lbd_sample) for lbd_sample in full_lbd_samples])
    S_true = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
    print(f"Sobol estimation with true probabilities: {S_true}")

    # list of cv estimation
    classic_cv_estimations = []
    mIS_cv_estimations = []
    # initial estimation and cv 
    p_f_and_cv = np.array([compute_pf_cv_alt(lbd_sample,g,X,fail_times,QoI_threshold) for lbd_sample in full_lbd_samples])
    p_f = p_f_and_cv[:,0]
    S_classic = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
    print(f"Sobol estimation with simple IS: {S_classic}")
    classic_cv_estimations += [np.max(p_f_and_cv[:,1])]

    p_f = [1 for _ in full_lbd_samples]
    cv = [np.inf for _ in full_lbd_samples]
    useful_X_index = [np.zeros(0) for _ in full_lbd_samples]
    cv_lists = [[] for _ in full_lbd_samples]
    # previous arg for cv_max for comparison, to avoid recomputation of AIS-CE in case of same index for cv_max
    cv_arg_max_previous = -1
    while cv_max>eps:
    #while n_IS<7:
        print(f"n_IS = {n_IS}")
        
        # compute p_T estimations for all lbd samples
        p_f_and_cv = np.array([compute_pf_cv_mult_alt(cv_lists[lbd_index],useful_X_index[lbd_index],p_f[lbd_index],cv[lbd_index],full_lbd_samples[lbd_index],g_list,X,fail_times,T,g_sum_array,f_array_list,lbd_index) for lbd_index in range(len(full_lbd_samples))])
        p_f = p_f_and_cv[:,0]
        #print(f"10 first estimated failure proba: {p_f[:10]}")
        S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
        print(f"current Sobol index estimation: {S}")
        cv = p_f_and_cv[:,1]
        # check first parameters with highest cv
        cv_argsort = np.argsort(cv)
        cv_arg_max = np.argmax(cv)
        cv_max = cv[cv_arg_max]
        lbd_max = full_lbd_samples[cv_arg_max]
        print(f"cv max: {cv_max}")
        print(f"5 highest cv: {[cv[cv_argsort[-i]] for i in range(1,6)]}")
        print(f"corresponding parameters: {[full_lbd_samples[cv_argsort[-i]] for i in range(1,6)]}")
        if cv_max > eps:
        # if cv_arg_max different from previous cv_arg_max, do AIS-CE, else 
            if cv_arg_max != cv_arg_max_previous:
                # default start point (not used, only for shape)
                lbd_start = np.array([[0.,10.] for _ in range(len(parameter_list))])
                # start with previous point               
                _,_,lbd_g_new,n_resample = AIS_CE(lbd_start,lbd_max,parameter_list,T,0,N,alpha,[i for i in range(len(parameter_list))])
                draw_g_new = [(lambda lbd_t=lbd_i: np.random.default_rng().normal(loc = lbd_t[0],scale=lbd_t[1])) for lbd_i in lbd_g_new]
                g_list+=[[(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g_new]]
                # draw samples according to the density
                X_new,_,fail_times_new = sample_IS(draw_g_new,parameter_list,N)

            else:
                print("lbd index yielding cv_max is the same as previously, reuse of previous AIS-CE results")
                g_list+=[[(lambda x, lbd_t=lbd_i: 1/(np.sqrt(2*np.pi)*lbd_t[1])*np.exp(-0.5*((x-lbd_t[0])/lbd_t[1])**2)) for lbd_i in lbd_g_new]]
            # update samples and g (multiple IS)
            X = np.concatenate((X,X_new))
            fail_times =np.concatenate((fail_times,fail_times_new))

            #lbd_start = lbd_g_new.copy()
            cv_arg_max_previous = cv_arg_max
            n_IS+=1

    
    S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)
    return
# def draw_X_IS(parameter_list,max_time,coeff):
#     """
#     Draw delays according to a density g 
#     that multiplies f by a coefficient >1 if x<=T, and accordingly multiplies f by a coefficient <1 if x>T
#     """
#     # sample delays according to modified g
#     delays = []
#     for tr in parameter_list:
#         # parameter of the f exponential law
#         lbd = tr[2]
#         # bernoulli parameter corresponds to the proportion of the integral of g coming from the part inferior to max_time
#         bernouilli = np.random.default_rng().binomial(1,1-(1-coeff)*np.exp(-lbd*max_time))
#         if bernouilli==1:
#             # inverse cdf to sample f between 0 and max_time
#             inverse_cdf = lambda t: -(np.log(1-t*(1-np.exp(-lbd*max_time)))/lbd)
#         else:
#             # inverse cdf to sample f between max_time and +infty
#             inverse_cdf = lambda t: max_time-(np.log(1-t)/lbd)
#         delays.append([tr[0],inverse_cdf(np.random.default_rng().random())])
#     return delays

# def simple_IS(parameter_list,max_time,coeff,obs_name,n_run):
#     """
#     Run an IS simulation of size n_run
#     Returns the drawn delays with the associated AltaRica simulation result

#     """
#     # drawn delays
#     X = np.zeros((n_run,len(parameter_list)))
#     # results of simulation
#     Y = np.zeros(n_run)

#     # multiplication factor (f/g)(x) for the IS
#     f_g = lambda lbd,x: 1/((x<=max_time)*(1+coeff*(np.exp(-lbd*max_time)/(1-np.exp(-lbd*max_time))))+(x>max_time)*(1-coeff))
#     # table for the computation of the multiplication factors
#     f_g_array = np.zeros(n_run)
#     # final IS probability
#     p_f = 0
#     for simu_index in range(n_run):
#         delays = draw_X_IS(parameter_list,max_time,coeff)
#         X[simu_index] = [couple[1] for couple in delays]
#         Y[simu_index] = step(delays,max_time,obs_name)
#         if Y[simu_index]==1:
#             # update IS sum
#             f_g_array[simu_index]=np.prod([f_g(parameter_list[tr_index][2],X[simu_index][tr_index]) for tr_index in range (len(parameter_list))])
#             """
#             for tr in range(len(parameter_list)):
#                 x,lbd = X[simu_index][tr],parameter_list[tr][2]
#                 print(f"x = {x}")
#                 print((x<=max_time)*(1+coeff*(np.exp(-lbd*max_time)/(1-np.exp(-lbd*max_time)))+(x>max_time)*(1-coeff)))
#             print(p_f)
#             """  
#     p_f=np.mean(f_g_array)

#     # CV estimation
#     cv = np.sqrt(np.var(f_g_array))/(np.sqrt(n_run)*p_f)
#     return X,Y,p_f,cv

def get_samples_fail(X,fail_times,obs_time):
    """
    Get samples leading to a failure at obs_time
    """
    N = len(X)
    # samples leading to failure
    samples = []
    for n in range(N):
        if fail_times[n]<obs_time:
            samples.append(X[n])
    return np.array(samples)


if __name__ == '__main__':


    
    X_m,Y_m,fail_times_m=Monte_Carlo(get_X_param(),10000) 
    print(f"failure proba: {np.sum(Y_m)/10000}")