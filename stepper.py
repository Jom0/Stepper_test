import subprocess
import os
import numpy as np
import time




"""
PATH = "D:\jo.mboko\Documents\Altarica\Step_test"

STEPPER = "gtsstp.exe"

# name of .gts 
EXPERIMENT_NAME = "step_test"
# name of observer
OBS_NAME = "not_Powered"

"""
PATH = "D:\jo.mboko\Documents\Altarica\Simple_ROSA"

STEPPER = "gtsstp.exe"

# name of .gts 
EXPERIMENT_NAME = "UAVFunctionalview"
# name of observer
OBS_NAME = "CAT_SOL"


os.chdir(PATH)

def get_transitions():
    # to do with XML file parsing
    """
    Get number, names and exponential parameter associated to each transition firable at the start

    For now we consider only the case following case: all the non-determinist transitions are firable at the start 
    and their delays are drawn according to an exponential law
    
    """
    transition_list = []

    # get firable transition list as txt file
    with open("tmp_script.txt","w") as tmp:    
        tmp.write("print tr\nquit")
    subprocess.call(f"{STEPPER} {EXPERIMENT_NAME}.gts --script tmp_script.txt -o tmp_transition_list.txt")
    os.remove("tmp_script.txt")

    # convert to python list
    with open("tmp_transition_list.txt",'r') as tr_file:
        for line in tr_file:
            if line.strip().startswith("["):    
                parts = line.split()
                index = int(parts[0][1:-1])  
                name = parts[1]  
                delay = float(parts[5])  
                transition_list.append([index, name, delay])
    os.remove("tmp_transition_list.txt")
    
    return transition_list
        
def draw_delay(transition_list):
    """
    Draw the delays associated to each transition of transition_list according to an exponential law of appropriate parameter
    This corresponds to one simulation
    """
    # list of couples of indices of transition associated with a delay
    delays = []
    for tr in transition_list:
        delays.append([tr[0],np.random.default_rng().exponential(scale=1/tr[2])])
    return delays


def step(delays,max_time,obs_name) :
    # could be modified to take into account multiple times and observers
    """
    Run the execution of the AltaRica simulation corresponding to delays until max_time
    and return the final obs_name value, and the precise time of failure of the system

    The observer takes boolean values so we translate "true" to 1 and "false" to 0
    """
    obs_value = 0
    # sort transitions in order of increasing delay
    sorted_transitions = sorted(delays,key=lambda x: x[1])

    # create script for the stepper
    with open("tmp_script.txt","w") as tmp:
        for transition in sorted_transitions:
            if transition[1]>max_time:
                break
            # fire earliest transition and propagate immediate transitions
            tmp.write(f"fire {transition[0]}\nfire -i\n")
            # check observer value
            tmp.write(f"print o\n")
            # debug
            #tmp.write("print v\nprint o\n")
        # get observers value at the end of the simulation and quit stepper
        tmp.write("save o tmp_obs.txt\nquit")
    # run stepper with script
    subprocess.call(f"{STEPPER} {EXPERIMENT_NAME}.gts --mode advanced --no-timed -O assertion --script tmp_script.txt -o full_obs.txt")
    os.remove("tmp_script.txt")

    # get observer value
    with open("tmp_obs.txt","r") as tmp:
        for line in tmp:
            if line.strip().startswith(obs_name):
                parts = line.split()
                if parts[2]=="true":
                    obs_value = 1
                elif parts[2]=="false":
                    obs_value = 0
                else:
                    raise ValueError("no boolean value found for observer")
                break
    os.remove("tmp_obs.txt")
    
    # get exact time of failure (if system failed, else output inf)
    with open("full_obs.txt","r") as tmp:
        tr_index = 0
        fail_time = np.inf
        for line in tmp:
            if line.strip().startswith(obs_name):
                parts = line.split()
                if parts[2]=="true":
                    fail_time = sorted_transitions[tr_index][1]
                    break
                tr_index+=1
    os.remove("full_obs.txt")
    return obs_value, fail_time

def Monte_Carlo(transition_list,max_time,obs_name,n_run):
    """
    Run a Monte-Carlo simulation of size n_run
    Returns the drawn delays with the associated AltaRica simulation result
    """
    # drawn delays
    X = np.zeros((n_run,len(transition_list)))
    # results of simulation
    Y = np.zeros(n_run)
    # fail times
    fail_times = np.zeros(n_run)
    for simu_index in range(n_run):
        delays = draw_delay(transition_list)
        X[simu_index] = [couple[1] for couple in delays]
        simu = step(delays,max_time,obs_name)
        Y[simu_index] = simu[0]
        fail_times[simu_index] = simu[1]
    return X,Y,fail_times

def time_execution(transition_list,max_time,obs_name,n_run):
    """
    Evaluate which part of the code takes the most time
    """
    def step_timed(delays,max_time,obs_name) :
        # could be modified to take into account multiple times and observers
        """
        Run the execution of the AltaRica simulation corresponding to delays until max_time
        and return the final obs_name value, and the precise time of failure of the system

        The observer takes boolean values so we translate "true" to 1 and "false" to 0
        """
        obs_value = 0
        # sort transitions in order of increasing delay
        sorted_transitions = sorted(delays,key=lambda x: x[1])
        start_time_script_1 = time.time()
        # create script for the stepper
        with open("tmp_script.txt","w") as tmp:
            for transition in sorted_transitions:
                if transition[1]>max_time:
                    break
                # fire earliest transition and propagate immediate transitions
                tmp.write(f"fire {transition[0]}\nfire -i\n")
                # check observer value
                tmp.write(f"print o\n")
                # debug
                #tmp.write("print v\nprint o\n")
            # get observers value at the end of the simulation and quit stepper
            tmp.write("save o tmp_obs.txt\nquit")
        start_time_stepper = time.time()
        # run stepper with script
        subprocess.call(f"{STEPPER} {EXPERIMENT_NAME}.gts --mode advanced --no-timed -O assertion --script tmp_script.txt -o full_obs.txt")
        os.remove("tmp_script.txt")
        start_time_script_2 = time.time()
        # get observer value
        with open("tmp_obs.txt","r") as tmp:
            for line in tmp:
                if line.strip().startswith(obs_name):
                    parts = line.split()
                    if parts[2]=="true":
                        obs_value = 1
                    elif parts[2]=="false":
                        obs_value = 0
                    else:
                        raise ValueError("no boolean value found for observer")
                    break
        os.remove("tmp_obs.txt")
        start_time_script_3 = time.time()
        # get exact time of failure (if system failed, else output inf)
        with open("full_obs.txt","r") as tmp:
            tr_index = 0
            fail_time = np.inf
            for line in tmp:
                if line.strip().startswith(obs_name):
                    parts = line.split()
                    if parts[2]=="true":
                        fail_time = sorted_transitions[tr_index][1]
                        break
                    tr_index+=1
        os.remove("full_obs.txt")
        end_time = time.time()
        return obs_value, fail_time,start_time_stepper-start_time_script_1,start_time_script_2-start_time_stepper,start_time_script_3-start_time_script_2,end_time-start_time_script_3
    start_time = time.time()
    # drawn delays
    X = np.zeros((n_run,len(transition_list)))
    # results of simulation
    Y = np.zeros(n_run)
    # fail times
    fail_times = np.zeros(n_run)
    stepper_script_creation_time = 0
    stepper_run_time = 0
    get_obs_script_time = 0
    get_fail_time_script_time = 0
    for simu_index in range(n_run):
        delays = draw_delay(transition_list)
        X[simu_index] = [couple[1] for couple in delays]
        simu = step_timed(delays,max_time,obs_name)
        Y[simu_index] = simu[0]
        fail_times[simu_index] = simu[1]  
        stepper_script_creation_time += simu[2] 
        stepper_run_time += simu[3]
        get_obs_script_time += simu[4]
        get_fail_time_script_time += simu[5]
    end_time = time.time()
    return end_time-start_time, stepper_script_creation_time, stepper_run_time, get_obs_script_time, get_fail_time_script_time


def draw_delay_IS_g(draw_g,transition_list):
    """
    Draw delays using a list of function that generates samples draw_g
    """
    # sample delays according to modified g
    delays = []
    for tr in transition_list:
        sample = draw_g[tr[0]]()
        delays.append([tr[0],sample])
    return delays

def sample_IS(draw_g,transition_list,max_time,obs_name,n_run):
    """
    Run an simulation of size n_run
    Sampling done along draw_g
    Returns the drawn delays with the associated AltaRica simulation result
    """
    # drawn delays
    X = np.zeros((n_run,len(transition_list)))
    # results of simulation
    Y = np.zeros(n_run)
    # fail times
    fail_times = np.zeros(n_run)
    for simu_index in range(n_run):
        delays = draw_delay_IS_g(draw_g,transition_list)
        X[simu_index] = [couple[1] for couple in delays]
        simu = step(delays,max_time,obs_name)
        Y[simu_index] = simu[0]
        fail_times[simu_index] = simu[1]
    return X,Y,fail_times    




def compute_pf_cv(g,transition_list,X,fail_times,obs_time):
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
            for tr_index in range(len(transition_list)):
                lbd = transition_list[tr_index][2]
                x = X[simu_index][tr_index]
                f_g*= (lbd*np.exp(-lbd*x)/g[transition_list[tr_index][0]](x))
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

def AIS_CE(lbd_0,transition_list,obs_name,T,max_time,N,alpha,transition_index_list):
    """
    AIS_CE for p
    """
    n_tr = len(transition_list)
    quant_index = int(np.ceil(alpha*N))
    lbd = lbd_0.copy()
    # temporary lbd
    lbd_temp = lbd_0.copy()
    # current threshold time
    t_m = max_time
    while t_m>T:
        print(f"t_m = {t_m}")
        # sampling density
        draw_f = [(lambda lbd_t=lbd_i: np.random.default_rng().exponential(scale=1/lbd_t)) for lbd_i in lbd]
        X_IS,_,fail_times_IS  = sample_IS(draw_f,transition_list,max_time,obs_name,N)
        # quantile of level alpha
        #print(fail_times_IS)
        #print(quant_index)
        sorted_fail_times_IS = sorted(fail_times_IS)
        t_alpha = sorted_fail_times_IS[quant_index]
        t_m = max(t_alpha,T)
        # if t_alpha is infinite, take all the non infinite values
        if t_m == np.inf:
            print("less samples leading to failure than the requested quantile")
            max_index = quant_index
            while max_index >= 0 and sorted_fail_times_IS[max_index] == np.inf:
                max_index-=1
            if max_index == 0:
                raise Exception("None of the drawn samples lead to the failure event")
            t_m = sorted_fail_times_IS[max_index]
        # ratio between f_x|lbd_0 and f_x|lbd
        W = lambda x,lbd_0_w,lbd_w : np.prod([(lbd_0_w[tr_index]/lbd_w[tr_index]) for tr_index in transition_index_list])*np.exp(np.sum([-x[tr_index]*((lbd_0_w[tr_index]-lbd_w[tr_index])) for tr_index in transition_index_list]))
        #W = lambda x,lbd_0,lbd : np.prod([(lbd[tr_index]/lbd_0[tr_index]) for tr_index in range(n_tr)])*np.exp(np.sum([-x[tr_index]*((lbd[tr_index]-lbd_0[tr_index])/(lbd[tr_index]*lbd_0[tr_index])) for tr_index in range(n_tr)]))
        for tr_index in transition_index_list:
            lbd_num = 0
            lbd_denom = 0
            for sample_index in range(N):
                if fail_times_IS[sample_index]<=t_m :
                    W_i = W(X_IS[sample_index],lbd_0,lbd)
                    lbd_num+=W_i
                    lbd_denom+=W_i*X_IS[sample_index][tr_index]
            lbd_temp[tr_index] = lbd_num/lbd_denom
        lbd = lbd_temp.copy()
        # estimate probability and cv
        W_array = np.zeros(N)
        for sample_index in range(N):
            if fail_times_IS[sample_index]<=T :
                W_array[sample_index]=W(X_IS[sample_index],lbd_0,lbd)
        p_T = np.mean(W_array)
        cv = np.sqrt(np.var(W_array))/(np.sqrt(N)*p_T)
        print(f"current probability estimation: {p_T}")
        print(f"current cv estimation: {cv}")
    # estimate probability and cv
    W_array = np.zeros(N)
    for sample_index in range(N):
        if fail_times_IS[sample_index]<=T :
            W_array[sample_index]=W(X_IS[sample_index],lbd_0,lbd)
    p_T = np.mean(W_array)
    if p_T>1:
        raise Exception("Final estimated probability is greater than 1")
    cv = np.sqrt(np.var(W_array))/(np.sqrt(N)*p_T)
        
    return p_T,cv,lbd


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

def Sobol(lbd_g,lbd_indices_v,draw_lbd,transition_list,obs_name,T,max_time,N,M):
    """
    Sobol indices of lambda of indices lbd_indices_v, with lamba drawn according to draw_lbd
    Initial sampling density lbd_g obtained for a lbd_0
    """
    # initial lbd for the sampling density
    #_,_,lbd_g = AIS_CE(lbd_0,transition_list,obs_name,T,max_time,N,alpha,[i for i in range(len(transition_list))])
    draw_g = [(lambda lbd_t=lbd_i: np.random.default_rng().exponential(scale=1/lbd_t)) for lbd_i in lbd_g]
    g = [(lambda x, lbd_t=lbd_i: lbd_t*np.exp(-lbd_t*x)) for lbd_i in lbd_g]
    # draw samples according to the density
    X,Y,fail_times = step.sample_IS(draw_g,transition_list,max_time,obs_name,N)
    
    # draw lbd samples for pick freeze
    lbd_samples = np.array([[draw_lbd[tr_index]() for tr_index in range(len(transition_list))] for m in range(M)])
    lbd_samples_v = np.array([[draw_lbd[tr_index]() for tr_index in range(len(transition_list))] for m in range(M)])
    for index in lbd_indices_v:
        lbd_samples_v[:,index] = lbd_samples[:,index].copy()
    full_lbd_samples = np.concatenate((lbd_samples,lbd_samples_v))

    p_f_and_cv = np.array([compute_pf_cv_alt(lbd_sample,g,X,fail_times,T) for lbd_sample in full_lbd_samples])
    p_f = p_f_and_cv[:,0]
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

    return S
def Total_Sobol(lbd_g,lbd_index,draw_lbd,transition_list,obs_name,T,max_time,N,M):
    # get all indices different from lbd_index 
    lbd_v_indices = [i for i in range(len(lbd_g)) if i!=lbd_index]
    return 1-Sobol(lbd_g,lbd_v_indices,draw_lbd,transition_list,obs_name,T,max_time,N,M)

def compute_pf_cv_mult(lbd_new,g_list,X,fail_times,obs_time):
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

def Sobol_mIS(lbd_g,lbd_indices_v,draw_lbd,transition_list,obs_name,T,max_time,N,M,alpha,eps):
    """
    Sobol indices of lambda of indices lbd_indices_v, with lamba drawn according to draw_lbd
    Initial sampling density lbd_g obtained for a lbd_0
    Computation with multiple importance sampling
    """
    # initial lbd for the sampling density
    #_,_,lbd_g = AIS_CE(lbd_0,transition_list,obs_name,T,max_time,N,alpha,[i for i in range(len(transition_list))])
    #print("initial IS parameters:")
    #for x_index in range(len(lbd_g)):
    #    print(f"{transition_list[x_index][1]} ({x_index}): {lbd_g[x_index]}")
    draw_g = [(lambda lbd_t=lbd_i: np.random.default_rng().exponential(scale=1/lbd_t)) for lbd_i in lbd_g]
    g = [(lambda x, lbd_t=lbd_i: lbd_t*np.exp(-lbd_t*x)) for lbd_i in lbd_g]
    # draw samples according to the density
    X,_,fail_times = step.sample_IS(draw_g,transition_list,max_time,obs_name,N)
    
    # for test
    #X_copy = X.copy()
    #fail_times_copy = fail_times.copy()
    # draw lbd samples for pick freeze
    lbd_samples = np.array([[draw_lbd[tr_index]() for tr_index in range(len(transition_list))] for m in range(M)])
    lbd_samples_v = np.array([[draw_lbd[tr_index]() for tr_index in range(len(transition_list))] for m in range(M)])
    for index in lbd_indices_v:
        lbd_samples_v[:,index] = lbd_samples[:,index].copy()
    full_lbd_samples = np.concatenate((lbd_samples,lbd_samples_v))

    # list of sampling functions obtained with AIS-CE
    g_list = [g]
    
    n_IS = 1
    cv_max = 1
    while cv_max>eps:
    #while n_IS<7:
        print(f"n_IS = {n_IS}")
        # compute p_T estimations for all lbd samples
        # to change to avoid recomputing failure proba and cv for same index
        p_f_and_cv = np.array([compute_pf_cv_mult(lbd_sample,g_list,X,fail_times,T) for lbd_sample in full_lbd_samples])
        p_f = p_f_and_cv[:,0]
        print(f"10 first estimated failure proba: {p_f[:10]}")
        cv = p_f_and_cv[:,1]
        cv_arg_max = np.argmax(cv)
        cv_max = cv[cv_arg_max]
        lbd_max = full_lbd_samples[cv_arg_max]
        print(f"cv max: {cv_max}")
        if cv_max > eps:
            _,_,lbd_g_new = AIS_CE(lbd_max,transition_list,obs_name,T,max_time,N,alpha,[i for i in range(len(transition_list))])
            #lbd_g_new = lbd_g.copy()
            #print("new IS parameters")
            #for x_index in range(len(lbd_g_new)):
            #    print(f"{transition_list[x_index][1]} ({x_index}): {lbd_g_new[x_index]}")
            draw_g_new = [(lambda lbd_t=lbd_i: np.random.default_rng().exponential(scale=1/lbd_t)) for lbd_i in lbd_g_new]
            g_list+=[[(lambda x, lbd_t=lbd_i: lbd_t*np.exp(-lbd_t*x)) for lbd_i in lbd_g_new]]
            # draw samples according to the density
            X_new,_,fail_times_new = step.sample_IS(draw_g_new,transition_list,max_time,obs_name,N)
            #X_new = X_copy.copy()
            #fail_times_new = fail_times_copy.copy()
            # update samples and g (multiple IS)
            X = np.concatenate((X,X_new))
            fail_times =np.concatenate((fail_times,fail_times_new))
            n_IS+=1

    
    S = ((1/M)*np.sum([p_f[i]*p_f[i+M]  for i in range(M)])-(1/M)*np.sum([p_f[i] for i in range(M)])* (1/M)*np.sum([p_f[i+M]  for i in range(M)]))/ ((1/M)*np.sum([p_f[i]**2 for i in range(M)])- ((1/M)*np.sum([p_f[i]  for i in range(M)]))**2)

    return S,g

# def draw_delay_IS(transition_list,max_time,coeff):
#     """
#     Draw delays according to a density g 
#     that multiplies f by a coefficient >1 if x<=T, and accordingly multiplies f by a coefficient <1 if x>T
#     """
#     # sample delays according to modified g
#     delays = []
#     for tr in transition_list:
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

# def simple_IS(transition_list,max_time,coeff,obs_name,n_run):
#     """
#     Run an IS simulation of size n_run
#     Returns the drawn delays with the associated AltaRica simulation result

#     """
#     # drawn delays
#     X = np.zeros((n_run,len(transition_list)))
#     # results of simulation
#     Y = np.zeros(n_run)

#     # multiplication factor (f/g)(x) for the IS
#     f_g = lambda lbd,x: 1/((x<=max_time)*(1+coeff*(np.exp(-lbd*max_time)/(1-np.exp(-lbd*max_time))))+(x>max_time)*(1-coeff))
#     # table for the computation of the multiplication factors
#     f_g_array = np.zeros(n_run)
#     # final IS probability
#     p_f = 0
#     for simu_index in range(n_run):
#         delays = draw_delay_IS(transition_list,max_time,coeff)
#         X[simu_index] = [couple[1] for couple in delays]
#         Y[simu_index] = step(delays,max_time,obs_name)
#         if Y[simu_index]==1:
#             # update IS sum
#             f_g_array[simu_index]=np.prod([f_g(transition_list[tr_index][2],X[simu_index][tr_index]) for tr_index in range (len(transition_list))])
#             """
#             for tr in range(len(transition_list)):
#                 x,lbd = X[simu_index][tr],transition_list[tr][2]
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
    # test execution time
    time_execution(get_transitions(),200,OBS_NAME,1000)

    
    #X_m,Y_m,fail_times_m=Monte_Carlo(transition_list,max_time,OBS_NAME,n_run) #failure proba for t=100 ~0.59 for test case
    #X,Y,p_f,c_v = simple_IS(transition_list,max_time,0.5,OBS_NAME,n_run)
    #print(f"failure proba at t={max_time}: {p_f}")
    #print(f"cv estimate for {n_run}: {c_v}")