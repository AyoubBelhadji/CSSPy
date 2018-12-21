
import scipy.io
import numpy as np
import pandas as pd

from oct2py import octave
# Generated with SMOP  0.41
from libsmop import *
# sRRQR_rank.m


#octave.addpath('/StrongRRQR')


#def Strong_RRQR(X,k):
#    Q, R, p = octave.sRRQR_rank(X,2,k)
#    return p[0:k]
        

    
def Strong_RRQR(A,k):
    f = 2
    varargin = sRRQR_rank.varargin
    nargin = sRRQR_rank.nargin

    
    #   Strong Rank Revealing QR with fixed rank 'k'
    
    #       A(:, p) = Q * R = Q [R11, R12; 
#                              0, R22]
#   where R11 and R12 satisfies that matrix (inv(R11) * R12) has entries
#   bounded by a pre-specified constant which should be not less than 1. 
#   
#   Input: 
#       A, matrix,  target matrix that is appoximated.
#       f, scalar,  constant that bound the entries of calculated (inv(R11) * R12)#    
#       k, integer, dimension of R11.
    
    #   Output: 
#       A(:, p) = [Q1, Q2] * [R11, R12; 
#                               0, R22]
#               approx Q1 * [R11, R12];
#       Only truncated QR decomposition is returned as 
#           Q = Q1, 
#           R = [R11, R12];
#       where Q is a m * k matrix and R is a k * n matrix
#   
#   Reference: 
#       Gu, Ming, and Stanley C. Eisenstat. "Efficient algorithms for 
#       computing a strong rank-revealing QR factorization." SIAM Journal 
#       on Scientific Computing 17.4 (1996): 848-869.
    
    #   Note: 
#       Algorithm 4 in the above ref. is implemented.
    
    #   check constant bound f
    if f < 1:
        fprintf('parameter f given is less than 1. Automatically set f = 2\n')
        f=2
# sRRQR_rank.m:39
    
    #   dimension of the given matrix
    m,n=size(A,nargout=2)
# sRRQR_rank.m:43
    #   modify rank k if necessary
    k=min(concat([k,m,n]))
# sRRQR_rank.m:46
    #   pivoting QR first (generally most time consuming step)
    Q,R,p=qr(A,0,nargout=3)
# sRRQR_rank.m:49
    #   check special case
    if (k == n):
        fprintf('Rank equals the number of columns!\n')
        return Q,R,p
    
    #   The following codes are the major part of the strong rank-revealing
#   algorithm which is based on the above pivoting QR. 
#   Name of variables are from the reference paper.
    
    #   make diagonals of R positive.
    if (size(R,1) == 1 or size(R,2) == 1):
        ss=sign(R(1,1))
# sRRQR_rank.m:64
    else:
        ss=sign(diag(R))
# sRRQR_rank.m:66
    
    R=bsxfun(times,R,reshape(ss,[],1))
# sRRQR_rank.m:68
    Q=bsxfun(times,Q,reshape(ss,1,[]))
# sRRQR_rank.m:69
    #   Initialization of A^{-1}B ( A refers to R11, B refers to R12)
    AB=linsolve(R(arange(1,k),arange(1,k)),R(arange(1,k),arange((k + 1),end())),struct('UT',true))
# sRRQR_rank.m:72
    #   Initialization of gamma, i.e., norm of C's columns (C refers to R22)
    gamma=zeros(n - k,1)
# sRRQR_rank.m:75
    if k != size(R,1):
        gamma=(sum(R(arange((k + 1),end()),arange((k + 1),end())) ** 2,1) ** (1 / 2)).T
# sRRQR_rank.m:77
    
    #   Initialization of omega, i.e., reciprocal of inv(A)'s row norm
    tmp=linsolve(R(arange(1,k),arange(1,k)),eye(k),struct('UT',true))
# sRRQR_rank.m:81
    omega=sum(tmp ** 2,2) ** (- 1 / 2)
# sRRQR_rank.m:82
    #   KEY STEP in Strong RRQR: 
#   "while" loop for interchanging the columns from first k columns and 
#   the remaining (n-k) columns.
    
    while 1:

        #   identify interchanging columns
        tmp=(dot(1.0 / omega,gamma.T)) ** 2 + AB ** 2
# sRRQR_rank.m:91
        i,j=find(tmp > dot(f,f),1,'first',nargout=2)
# sRRQR_rank.m:92
        #   present Q, R, p is a strong RRQR.
        if isempty(i):
            break
        #     fprintf('interchanging\n');
        #   Interchange the i th and (k+j) th column of target matrix A and 
    #   update QR decomposition (Q, R, p), AB, gamma, and omega.
    ##   First step : interchanging the k+1 and k+j th columns
        if j > 1:
            AB[arange(),concat([1,j])]=AB(arange(),concat([j,1]))
# sRRQR_rank.m:106
            gamma[concat([1,j])]=gamma(concat([j,1]))
# sRRQR_rank.m:107
            R[arange(),concat([k + 1,k + j])]=R(arange(),concat([k + j,k + 1]))
# sRRQR_rank.m:108
            p[concat([k + 1,k + j])]=p(concat([k + j,k + 1]))
# sRRQR_rank.m:109
        ##   Second step : interchanging the i and k th columns
        if i < k:
            p[arange(i,k)]=p(concat([arange((i + 1),k),i]))
# sRRQR_rank.m:114
            R[arange(),arange(i,k)]=R(arange(),concat([arange((i + 1),k),i]))
# sRRQR_rank.m:115
            omega[arange(i,k)]=omega(concat([arange((i + 1),k),i]))
# sRRQR_rank.m:116
            AB[arange(i,k),arange()]=AB(concat([arange((i + 1),k),i]),arange())
# sRRQR_rank.m:117
            for ii in arange(i,(k - 1)).reshape(-1):
                G=givens(R(ii,ii),R(ii + 1,ii))
# sRRQR_rank.m:120
                if dot(G(1,arange()),concat([[R(ii,ii)],[R(ii + 1,ii)]])) < 0:
                    G=- G
# sRRQR_rank.m:122
                R[arange(ii,ii + 1),arange()]=dot(G,R(arange(ii,ii + 1),arange()))
# sRRQR_rank.m:124
                Q[arange(),arange(ii,ii + 1)]=dot(Q(arange(),arange(ii,ii + 1)),G.T)
# sRRQR_rank.m:125
            if R(k,k) < 0:
                R[k,arange()]=- R(k,arange())
# sRRQR_rank.m:128
                Q[arange(),k]=- Q(arange(),k)
# sRRQR_rank.m:129
        ##   Third step : zeroing out the below-diag of k+1 th columns
        if k < size(R,1):
            for ii in arange((k + 2),size(R,1)).reshape(-1):
                G=givens(R(k + 1,k + 1),R(ii,k + 1))
# sRRQR_rank.m:136
                if dot(G(1,arange()),concat([[R(k + 1,k + 1)],[R(ii,k + 1)]])) < 0:
                    G=- G
# sRRQR_rank.m:138
                R[concat([k + 1,ii]),arange()]=dot(G,R(concat([k + 1,ii]),arange()))
# sRRQR_rank.m:140
                Q[arange(),concat([k + 1,ii])]=dot(Q(arange(),concat([k + 1,ii])),G.T)
# sRRQR_rank.m:141
        ##   Fourth step : interchaing the k and k+1 th columns
        p[concat([k,k + 1])]=p(concat([k + 1,k]))
# sRRQR_rank.m:146
        ga=R(k,k)
# sRRQR_rank.m:147
        mu=R(k,k + 1) / ga
# sRRQR_rank.m:148
        if k < size(R,1):
            nu=R(k + 1,k + 1) / ga
# sRRQR_rank.m:150
        else:
            nu=0
# sRRQR_rank.m:152
        rho=sqrt(dot(mu,mu) + dot(nu,nu))
# sRRQR_rank.m:154
        ga_bar=dot(ga,rho)
# sRRQR_rank.m:155
        b1=R(arange(1,(k - 1)),k)
# sRRQR_rank.m:156
        b2=R(arange(1,(k - 1)),k + 1)
# sRRQR_rank.m:157
        c1T=R(k,arange((k + 2),end()))
# sRRQR_rank.m:158
        c2T=R(k + 1,arange((k + 2),end()))
# sRRQR_rank.m:159
        c1T_bar=(dot(mu,c1T) + dot(nu,c2T)) / rho
# sRRQR_rank.m:160
        c2T_bar=(dot(nu,c1T) - dot(mu,c2T)) / rho
# sRRQR_rank.m:161
        R[arange(1,(k - 1)),k]=b2
# sRRQR_rank.m:164
        R[arange(1,(k - 1)),k + 1]=b1
# sRRQR_rank.m:165
        R[k,k]=ga_bar
# sRRQR_rank.m:166
        R[k,k + 1]=dot(ga,mu) / rho
# sRRQR_rank.m:167
        R[k + 1,k + 1]=dot(ga,nu) / rho
# sRRQR_rank.m:168
        R[k,arange((k + 2),end())]=c1T_bar
# sRRQR_rank.m:169
        R[k + 1,arange((k + 2),end())]=c2T_bar
# sRRQR_rank.m:170
        u=linsolve(R(arange(1,k - 1),arange(1,k - 1)),b1,struct('UT',true))
# sRRQR_rank.m:173
        u1=AB(arange(1,k - 1),1)
# sRRQR_rank.m:174
        AB[arange(1,k - 1),1]=(dot(dot(nu,nu),u) - dot(mu,u1)) / rho / rho
# sRRQR_rank.m:175
        AB[k,1]=mu / rho / rho
# sRRQR_rank.m:176
        AB[k,arange(2,end())]=c1T_bar / ga_bar
# sRRQR_rank.m:177
        AB[arange(1,k - 1),arange(2,end())]=AB(arange(1,k - 1),arange(2,end())) + (dot(dot(nu,u),c2T_bar) - dot(u1,c1T_bar)) / ga_bar
# sRRQR_rank.m:178
        gamma[1]=dot(ga,nu) / rho
# sRRQR_rank.m:181
        gamma[arange(2,end())]=(gamma(arange(2,end())) ** 2 + (c2T_bar.T) ** 2 - (c2T.T) ** 2) ** (1 / 2)
# sRRQR_rank.m:182
        u_bar=u1 + dot(mu,u)
# sRRQR_rank.m:185
        omega[k]=ga_bar
# sRRQR_rank.m:186
        omega[arange(1,k - 1)]=(omega(arange(1,k - 1)) ** (- 2) + u_bar ** 2 / (dot(ga_bar,ga_bar)) - u ** 2 / (dot(ga,ga))) ** (- 1 / 2)
# sRRQR_rank.m:187
        Gk=concat([[mu / rho,nu / rho],[nu / rho,- mu / rho]])
# sRRQR_rank.m:190
        if k < size(R,1):
            Q[arange(),concat([k,k + 1])]=dot(Q(arange(),concat([k,k + 1])),Gk.T)
# sRRQR_rank.m:192

    
    #   Only return the truncated version of the strong RRQR decomposition
    Q=Q(arange(),arange(1,k))
# sRRQR_rank.m:197
    R=R(arange(1,k),arange())
# sRRQR_rank.m:198
    return Q,R,p
    
if __name__ == '__main__':
    pass
    