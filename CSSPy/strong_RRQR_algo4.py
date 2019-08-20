
import scipy.io
import numpy as np
from math import hypot
from scipy import random, linalg, dot, diag, all, allclose
from copy import deepcopy
from scipy.linalg import solve_triangular
#data = scipy.io.loadmat("StrongRRQR/A.mat")
#
A = data["A"]

def givens_rotation_matrix_2(a, b):
    """Compute matrix entries for Givens rotation."""
    G=np.eye(2)
    r = hypot(a, b)
    if r == 0:
        c = 1
        s = 0
    else:
        c = a/r
        s = -b/r
    G[0,0] = c
    G[1,1] = c
    G[0,1] = -s
    G[1,0] = s

    return G

def permutation_matrix(d,i,j):
    perm_matrix = np.eye(d)
    perm_matrix[i,i] = 0
    perm_matrix[j,j] = 0
    perm_matrix[i,j] = 1
    perm_matrix[j,i] = 1
    return perm_matrix

def get_transposition_list(d,i,j):
    perm = list(range(d))
    perm[i] = j
    perm[j] = i
    return perm

def get_cyclic_permutation_list(d,i,j):
    perm = list(range(d))
    tmp = perm[i]
    for t in list(range(i,j)):
        perm[t] = perm[t+1]
    perm[j] = tmp
    return perm


def Strong_RRQR(A,k,f):

#    #   Strong Rank Revealing QR with fixed rank 'k'
#    
#    #       A(:, p) = Q * R = Q [R11, R12; 
##                              0, R22]
##   where R11 and R12 satisfies that matrix (inv(R11) * R12) has entries
##   bounded by a pre-specified constant which should be not less than 1. 
##   
##   Input: 
##       A, matrix,  target matrix that is appoximated.
##       f, scalar,  constant that bound the entries of calculated (inv(R11) * R12)#    
##       k, integer, dimension of R11.
#    
#    #   Output: 
##       A(:, p) = [Q1, Q2] * [R11, R12; 
##                               0, R22]
##               approx Q1 * [R11, R12];
##       Only truncated QR decomposition is returned as 
##           Q = Q1, 
##           R = [R11, R12];
##       where Q is a m * k matrix and R is a k * n matrix
##   
##   Reference: 
##       Gu, Ming, and Stanley C. Eisenstat. "Efficient algorithms for 
##       computing a strong rank-revealing QR factorization." SIAM Journal 
##       on Scientific Computing 17.4 (1996): 848-869.
#    
#    #   Note: 
##       Algorithm 4 in the above ref. is implemented.
#    

#   dimension of the given matrix
    m,n=np.shape(A)

    Q,R,p= linalg.qr(A, mode="full",pivoting=True)
    print(p)
    #print(R[0,0])
    print(p[0:10])
    #print(np.shape(Q))
    s_R = np.sign(np.diag(R))
    #print(s_R[0])
    for i in list(range(n)):
        R[i,i] = s_R[i]*R[i,i]
        Q[i,i] = s_R[i]*Q[i,i]
    
    #   Initialization of A^{-1}B ( A refers to R11, B refers to R12)
    R11 = deepcopy(R[0:k,0:k])
    R12 = deepcopy(R[0:k,k:])
    R22 = deepcopy(R[k:,k:])
    AB = deepcopy(np.dot(np.linalg.inv(R11),R12))
    #AB = solve_triangular(R11,R12)
    #print("AB")
    #print(np.amax(AB))
    #print(np.shape(AB))
    #print("ga11")
    #print(R[k-1,k-1])
    #   Initialization of gamma, i.e., norm of C's columns (C refers to R22)
    gamma = np.transpose(np.sqrt(np.diag(np.dot(np.transpose(R22),R22))))
    #print("gamma")
    #print(np.shape(gamma))
    #   Initialization of omega, i.e., reciprocal of inv(A)'s row norm
    tmp = np.linalg.pinv(R11)
    
    omega = 1./np.sqrt(np.diag(np.dot(tmp,np.transpose(tmp))))
    #print("omega")
    #print(np.shape(omega))
    #print(omega)

##   "while" loop for interchanging the columns from first k columns and 
##   the remaining (n-k) columns.
#    
    counter = 0
    while 1:
        tmp2 = np.power(np.outer(1./omega,np.transpose(gamma)),2) + np.power(AB,2)
        #print("tmp2")
        #print(np.shape(tmp2))
        #print(p[0:k])
        i_,j_ = np.where( tmp2 > np.power(f,2))
        print("size")
        print(i_.size)
        ind = np.unravel_index(np.nanargmax(tmp2, axis=None), tmp2.shape)
        i = ind[0]
        j = ind[1]
        print("max tmp2")
        print(tmp2[i,j])
        #if tmp2[i,j] <= np.power(f,2):
        #    break
        #if i_.size>0 and j_.size>0:
        #    print("yes")
        #    i = i_[0]
        #    j = j_[0]
        #else:
        #    break
        counter = counter +1
        #print("counter")
        #print(counter)
        print("AB")
        print(np.amax(AB))
        
#    Interchange the i th and (k+j) th column of target matrix A and 
#    update QR decomposition (Q, R, p), AB, gamma, and omega.
##   First step : interchanging the k+1 and k+j th columns

        if j > 0:
            #AB[:, [0, j]] = AB[:, [j, 0]]
            AB_d_0,_ = np.shape(AB)
            perm_AB_0 = get_transposition_list(AB_d_0,0,j)
            AB = AB[perm_AB_0]
            #gamma[[0, j]] = gamma[[j, 0]]
            gamma_tmp = gamma[0]
            gamma[0] = gamma[j]
            gamma[j] = gamma_tmp
            _,R_d_2 = np.shape(R)
            perm_R_1 = get_transposition_list(R_d_2,k,k+j-1)
            R = R[:,perm_R_1]
            #print("ga22")
            #print(R[k-1,k-1])
            #R[:, [k, k+j-1]] =R[:, [k+j-1, k]]
            #p[[k, k+j-1]] = p[[k+j-1, k]]
            p_tmp = p[k+j-1]
            p[k+j-1] = p[k]
            p[k] = p_tmp
##   Second step : interchanging the i and k th columns
        if i < k:
            _,R_d_2 = np.shape(R)
            perm_R_2 = get_cyclic_permutation_list(R_d_2,i,k-1)
            p = p[perm_R_2]

            R = R[:,perm_R_2]
            #print("ga33")
            #print(R[k-1,k-1])
            omega_d_1 = np.shape(omega)[0]
            perm_omega_3 = get_cyclic_permutation_list(omega_d_1,i,k-1)
            omega =  omega[perm_omega_3]
            
            AB_d_1,_ = np.shape(AB)
            perm_AB_3 = get_transposition_list(AB_d_1,i,k-1)

            #print(omega)

            AB =  AB[perm_AB_3, :]
            #print(AB)
#            %   givens rotation for the triangulation of R(1:k, 1:k)
            for ii in list(range(i,k)):
                G = givens_rotation_matrix_2(R[ii, ii], R[ii+1, ii])
                if np.dot(G[0,:],R[ii:ii+2,ii]) < 0:
                    G = -G  #  guarantee R(ii,ii) > 0
                    print("ok")
                R[ii:ii+2, :] = np.dot(G, R[ii:ii+2, :])
                Q[:, ii:ii+2] = np.dot(Q[:, ii:ii+2], np.transpose(G))
                
            if R[k-1,k-1] < 0:
                #print("ok")
                R[k-1, :] = - R[k-1, :]
                Q[:, k-1] = -Q[:, k-1]


##   Third step : zeroing out the below-diag of k+1 th columns
        R_m,R_n = np.shape(R)

        if k < R_m:
            for ii in list(range(k+1,R_m)):
                G=givens_rotation_matrix_2(R[k,k],R[ii,k])

                R_vstack = np.transpose(np.asarray([R[k, k],R[ii, k]]))
                if np.dot(G[0,:],R_vstack) < 0:
                    G = -G     #%  guarantee R(k+1,k+1) > 0
                _,R_d_4 = np.shape(R)
                #perm_R_4 = get_transposition_list(R_d_4,k,ii)                
                R[[k, ii], :] = np.dot(G,R[[k, ii], :])
                Q[:,[k, ii]] = np.dot(Q[:, [k, ii]],np.transpose(G))
##   Fourth step : interchaing the k and k+1 th columns
        #p[[k-1,k]] = p[[k, k-1]]
        p_tmp = p[k-1]
        p[k-1] = p[k]
        p[k] = p_tmp
        ga = deepcopy(R[k-1,k-1])
        mu = deepcopy(R[k-1,k])/ga
        if k < R_m:
            nu = deepcopy(R[k,k])/ga
        else:
            nu = 0
        rho = np.sqrt(mu*mu+nu*nu)
        ga_bar = ga*rho
        b1 = R[0:k-1,k-1]
        b2 = R[0:k-1,k]
        c1T = R[k-1,k+1:]
        c2T = R[k,k+1:] 
        #print(R[0,0])
        c1T_bar=(mu*c1T + nu*c2T)/rho
        c2T_bar=(nu*c1T - mu*c2T)/rho

        R[0:k-1,k-1]=b2
        R[0:k-1,k]=b1
        R[k-1,k-1]=ga_bar
        R[k-1,k]=np.dot(ga,mu)/rho
        R[k,k]=np.dot(ga,nu)/rho
        R[k-1,k+1:]=c1T_bar
        R[k,k+1:]=c2T_bar

        R_submatrix_tmp = deepcopy(R[0:k-1,0:k-1])
        u = np.dot(np.linalg.pinv(R_submatrix_tmp),b1)
        u1 = AB[0:k-1,0]
        AB[0:k-1,0] = ((mu*mu)*u - mu*u1)/(rho*rho)
        AB[k-1,0] = mu/(rho*rho)
        AB[k-1,1:] = c1T_bar/ga_bar        
        AB[0:k-1, 1:] = AB[0:k-1, 1:] + (nu*np.outer(u,c2T_bar) - np.outer(u1,c1T_bar))/ga_bar
        gamma[0] = ga*nu/rho
        gamma[1:] = np.power((np.power(gamma[1:],2) + np.power(np.transpose(c2T_bar),2) - np.power(np.transpose(c2T),2)),1/2)

        u_bar = u1 + mu*u
        omega[k-1] = ga_bar
        #print(np.power(omega[0:k-1],(-2)))
        #print("ga_bar")
        #print(ga_bar)
        #print(mu)
        #print("mu")
        #print(np.shape(omega))
        if counter ==0:
            omega[0:k-1] = np.power(np.abs((np.power(omega[0:k-1],(-2)) + np.power(u_bar,2)/(ga_bar*ga_bar) - np.power(u,2)/(ga*ga))),(-1/2))
        else:
            omega[0:k-1] = np.power((np.power(omega[0:k-1],(-2)) + np.power(u_bar,2)/(ga_bar*ga_bar) - np.power(u,2)/(ga*ga)),(-1/2))
        #print("counter")
        #print(counter)
        print(p[0:20])
#Eliminate new R(k+1, k) by orthgonal transformation

        Gk = np.asarray([[mu/rho,nu/rho],[nu/rho,-mu/rho]])
        #print(np.dot(Gk,np.transpose(Gk)))
        R_d_final,_ = np.shape(R)
        if k < R_d_final:
            Q[:, [k,k+1]] = np.dot(Q[:, [k,k+1]], np.transpose(Gk))
        

#    
    return p[0:k]
#A = np.transpose(r_X)
l = Strong_RRQR(A,200,1.05)