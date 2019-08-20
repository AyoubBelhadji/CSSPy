function [Q,R,p] = sRRQR(A, f, type, par)
%   
%   Strong Rank Revealing QR   
%       A P = [Q1, Q2] * [R11, R12; 
%                           0, R22]
%   where R11 and R12 satisfy that (inv(R11) * R12) has entries
%   bounded by a pre-specified constant which is not less than 1. 
%   
%   Input: 
%       A, matrix, target matrix that is appoximated.
%       f, scalar, constant that bounds the entries of calculated (inv(R11) * R12)
%    type, string, be either "rank" or "tol" and specify the way to decide
%                  the dimension of R11, i.e., the truncation. 
%     par, scalar, the parameter for "rank" or "tol" defined by the type. 
%
%   Output: 
%       A(:, p) = [Q1, Q2] * [R11, R12; 
%                               0, R22]
%               approx Q1 * [R11, R12];
%       Only truncated QR decomposition is returned as 
%           Q = Q1, 
%           R = [R11, R12];
%   
%   Reference: 
%       Gu, Ming, and Stanley C. Eisenstat. "Efficient algorithms for 
%       computing a strong rank-revealing QR factorization." SIAM Journal 
%       on Scientific Computing 17.4 (1996): 848-869.
%
%   Note: 
%       1. For a given rank (type = 'rank'), algorithm 4 in the above ref.
%       is implemented.
%       2. For a given error threshold (type = 'tol'), algorithm 6 in the
%       above ref. is implemented. 


%   given a fixed rank 
if (strcmp(type, 'rank'))
    [Q,R,p] = sRRQR_rank(A, f, par);
    return ;
end

%   given a fixed error threshold
if (strcmp(type, 'tol'))
    [Q,R,p] = sRRQR_tol(A, f, par);
    return ;
end

%   report error otherwise
fprintf('No parameter type of %s !\n', type)
Q = [];
R = [];
p = [];

end