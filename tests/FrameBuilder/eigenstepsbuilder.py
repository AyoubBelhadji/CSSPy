import numpy as np

def get_eigensteps_random(mu_vector,lambda_vector,N,d):
    ''' Construct a valid random point in the GT polytope.
    Parameters
    ----------
    mu_vector : array_type
                The vector of lengths
    lambda_vector : array_type
                The vector of spectrum
    N : int
        The number of elements in the frame
    d : int
        The dimension of the vector space
    Returns
    -------
    E : array_type
    '''
    E = np.zeros((N,N))
    E[:,N-1] = lambda_vector
    for n in range(N-2,-1,-1):
        for k in range(n, -1, -1):
            A_n_1_k = max(E[k+1,n+1],np.sum(E[k:n+2,n+1])-np.sum(E[k+1:n+1,n])-mu_vector[n+1])
            B_array = np.zeros(k+1)
            for l in range(k+1):
                B_array[l] = np.sum(mu_vector[l:n+1])-np.sum(E[l+1:k+1,n+1])-np.sum(E[k+1:n+1,n])
            B_n_1_k = min(E[k,n+1],min(B_array))
            if B_n_1_k<A_n_1_k:
                E[k,n] = B_n_1_k
            else:		
                u = np.random.uniform(0,1)
                delta_n_1_k = B_n_1_k - A_n_1_k
                E[k,n] = A_n_1_k + u*delta_n_1_k
            #print(A_n_1_k)
            #print(B_n_1_k)		
    return E

def get_eigensteps_mean(mu_vector,lambda_vector,N,d):
    ''' Construct a valid point in the GT polytope.
    Parameters
    ----------
    mu_vector : array_type
                The vector of lengths
    lambda_vector : array_type
                The vector of spectrum
    N : int
        The number of elements in the frame
    d : int
        The dimension of the vector space
    Returns
    -------
    E : array_type
    '''
    E = np.zeros((N,N))
    E[:,N-1] = lambda_vector
    for n in range(N-2,-1,-1):
        for k in range(n, -1, -1):
            A_n_1_k = max(E[k+1,n+1],np.sum(E[k:n+2,n+1])-np.sum(E[k+1:n+1,n])-mu_vector[n+1])
            B_array = np.zeros(k+1)
            for l in range(k+1):
                B_array[l] = np.sum(mu_vector[l:n+1])-np.sum(E[l+1:k+1,n+1])-np.sum(E[k+1:n,n])
            B_n_1_k = min(E[k,n+1],min(B_array))
            u = np.random.uniform(0,1)
            E[k,n] = A_n_1_k
    return E

def get_index_lists_I_and_J(E,n,N,d):
    ''' Construct the subsets I_n and J_n
    Parameters
    ----------
    E : array_type
        The matrix of eigensteps
    n : int
        The level of eigenvalues
    N : int
        The number of elements in the frame
    d : int
        The dimension of the vector space
    Returns
    -------
    I_n : list
    J_n : list
    '''
    I_n = list(range(d))
    J_n = list(range(d))
    n_ = n+1
    for m in reversed(range(d)):
        if E[m,n_-1] in E[J_n,n_]:
            del I_n[m]
            t_J_n = [i for i in J_n if E[i,n_] == E[m,n_-1] ]
            m_max = max(t_J_n)
            del J_n[m_max]
    return I_n,J_n

def diff_of_lists(first, second):
    ''' Construct the difference between two lists
    Parameters
    ----------
    first : list
        The first list
    second : list
        The second list
    '''
    second = set(second)
    return [item for item in first if item not in second]

def get_permutation_I(I_n,d):
    ''' Construct the permutation from I_n or J_n
    Parameters
    ----------
    I_n : list
        The list I_n
    d : int
        The dimension of the vector space
    Returns
    -------
    permutation : list
    '''
    permutation = [0]*d
    r_n = np.shape(I_n)[0]
    complementary_I_n = diff_of_lists(list(range(d)),I_n)
    c_r_n = d-r_n
    for i in range(r_n):
        permutation[I_n[i]] = i
    for i in range(c_r_n):
        permutation[complementary_I_n[i]] = i + r_n
    return permutation


def get_v_n_w_n(E,I_n,J_n,d,n):
    r_n = np.shape(I_n)[0]
    v_n = np.zeros((r_n))
    w_n = np.zeros((r_n))
    permutation_I_n = get_permutation_I(I_n,d)
    permutation_J_n = get_permutation_I(J_n,d)
    for m in I_n:
        v_n_index = permutation_I_n[m]
        nom_v_n = E[m,n]*np.ones((r_n,1)) - E[list(J_n),n+1]
        I_n_without_m = diff_of_lists(I_n,[m])
        cardinal_I_n_without_m = np.shape(I_n_without_m)[0]
        denom_v_n = E[m,n]*np.ones((cardinal_I_n_without_m,1)) - E[list(I_n_without_m),n]
        v_n[v_n_index] = np.sqrt(-np.prod(nom_v_n)/np.prod(denom_v_n))
    for m in J_n:
        w_n_index = permutation_J_n[m]
        nom_w_n = E[m,n+1]*np.ones((r_n,1)) - E[list(I_n),n]
        J_n_without_m = diff_of_lists(J_n,[m])
        cardinal_J_n_without_m = np.shape(J_n_without_m)[0]
        denom_w_n = E[m,n+1]*np.ones((cardinal_J_n_without_m,1)) - E[list(J_n_without_m),n+1]
        w_n[w_n_index] = np.sqrt(np.prod(nom_w_n)/np.prod(denom_w_n))
        
    return v_n,w_n



def get_permutation_matrix(permutation,d):
    permutation_matrix = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if i == permutation[j]:
                permutation_matrix[i,j] = 1
    return permutation_matrix



def get_W_n_matrix(E,I_n,J_n,d,n):
    r_n = np.shape(I_n)[0]
    v_n,w_n = get_v_n_w_n(E,I_n,J_n,d,n)
    W_n_matrix = np.zeros((r_n,r_n))
    permutation_I_n = get_permutation_I(I_n,d)
    permutation_J_n = get_permutation_I(J_n,d)
    for m in I_n:
        for m_ in J_n:
            v_n_index = permutation_I_n[m]
            w_n_index = permutation_J_n[m_]
            W_n_matrix[v_n_index,w_n_index] = 1/(E[m_,n+1]-E[m,n])*v_n[v_n_index]*w_n[w_n_index]
    return W_n_matrix


def get_padded_vector(v,d):
    r_n = np.shape(v)[0]
    v_padded = np.zeros((d,))
    v_padded[0:r_n] = v
    return v_padded



def get_extended_matrix_W(W_n_matrix,d):
    r_n = np.shape(W_n_matrix)[0]
    W_extended = np.eye(d)
    W_extended[0:r_n,0:r_n] = W_n_matrix
    return W_extended

def generate_random_diagonal_unimodular_matrix(d):
    vector = np.ones((1,d)) - 2*np.random.binomial(1, 0.5, d)
    return np.diag(vector[0])

def get_F_n_U_n(n,d,N,E,mu_vector,U_n_1):
    if n==1:
        M = np.eye(d)
        v = np.sqrt(mu_vector[0])*M[:,0]
        return v,np.eye(d)
    I_n,J_n = get_index_lists_I_and_J(E,n-1,N,d)
    r_n = np.shape(I_n)[0]
    permutation_matrix_I_n = get_permutation_matrix(get_permutation_I(I_n,d),d)
    permutation_matrix_J_n = get_permutation_matrix(get_permutation_I(J_n,d),d)
    v_n,w_n = get_v_n_w_n(E,I_n,J_n,d,n-1)
    W_extended = get_extended_matrix_W(get_W_n_matrix(E,I_n,J_n,d,n-1),d)
    v_padded = get_padded_vector(v_n,d)
    V_n = generate_random_diagonal_unimodular_matrix(d)
    U_n = np.dot(np.dot(np.dot(np.dot(U_n_1,V_n),np.transpose(permutation_matrix_I_n)),W_extended),permutation_matrix_J_n)
    f_n = np.dot(np.dot(np.dot(U_n_1,V_n),np.transpose(permutation_matrix_I_n)),v_padded)
    return f_n,U_n

def get_F(d,N,E,mu_vector):
    F_test = np.zeros((d,N))
    U_n_1 = np.eye(d)
    for n in range(N):
        #print(n)
        if n ==0:
            M = np.eye(d)
            F_test[:,n],U_n_1 = get_F_n_U_n(n+1,d,N,E,mu_vector,M)
        else:
            F_test[:,n],U_n_1 = get_F_n_U_n(n+1,d,N,E,mu_vector,U_n_1)
    return F_test

