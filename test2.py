def best_combination_estimation_ideal(true_pf,error_list,p_f_previous,var_previous,error_previous,f_array,g_array_list,g_list_duplicates,N):
    """
    Find combination of samples that yields the estimation of p_f for a given lbd with the smallest cv
    f_g_array array of the f/g coefficient for the lbd of interest
    """
    n_total_sample = len(f_array)
    # estimation for last sample only
    f_g = f_array[n_total_sample-N:]/g_array_list[-1][n_total_sample-N:]
    p_f_last = np.mean(f_g)
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
    current_error = error_list[-1]
    current_var = np.var(f_g)
    

    g = g_array_list[useful_g_index].copy()
    g_candidate = g.copy()
    #print(cv_list)
    #print(useful_g_index)
    #print(f"len of f_array: {len(f_array)}")
    #print(f"len of f_array slice: {len(f_array[useful_g_index*N:(useful_g_index+1)*N])}")
    #print(f"len of g: {len(g)}")
    p_f = np.mean(f_array[useful_g_index*N:(useful_g_index+1)*N]/g[useful_g_index*N:(useful_g_index+1)*N])

    # go through increasing cv
    #argsorted_var_list = np.argsort(var_list)
    argsorted_error_list = np.argsort(error_list)
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

        f_g = f_array[candidate_X_indices[:(n_used_indices+1)*N]]/g_candidate[candidate_X_indices[:(n_used_indices+1)*N]]

        p_f_test = np.mean(f_g)
        # compute variance of mIS estimator
        #var = (current_var*(n_used_indices)**2/(n_used_indices+1)**2)+np.var(f_array[index*N:(index+1)*N]/g_candidate[index*N:(index+1)*N])/(N*(n_used_indices+1)**2)
        var = 0
        for i in used_indices+[index]:
            var+=np.var(f_array[i*N:(i+1)*N]/g_candidate[i*N:(i+1)*N])
            
        var/=((n_used_indices+1)**2*N)
        cv = np.sqrt(var)/p_f_test

        error =  np.abs(p_f_test/true_pf-1)

        #if var<current_var:
        if error<current_error:
            #useful_X_index[n_used_indices*N:(n_used_indices+1)*N] = candidate_X_indices[n_used_indices*N:(n_used_indices+1)*N]
            g = g_candidate.copy()
            current_cv = cv
            current_var = var
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
    #if current_var<var_previous:
    if current_error<error_previous:
        return p_f,current_error,current_var,False
    else:
        #if np.abs(current_var-var_previous)/np.min([current_var,var_previous])>var_previous:
        if np.abs(current_error-error_previous)/np.min([current_error,error_previous])>error_previous:
            print("PROBLEM: new computed error superior to previous error")
            #print(f"previous cv : {cv_previous}, current cv : {current_cv}")
            print(f"previous error : {error_previous}, current error : {current_error}")
            print(f"previous var : {var_previous}, current var : {current_var}")
    return p_f_previous,error_previous,var_previous,True
