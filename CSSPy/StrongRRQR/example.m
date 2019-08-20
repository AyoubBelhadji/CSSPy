%%  Numerically low-rank matrix example
%   two well-separated point clusters
X0 = 4 * (1 - 2 *rand(2000,3));
Y0 = bsxfun(@plus, [12, 0, 0], 4 * (1-2*rand(2000, 3)));

%   kernel block defined by K(x,y) = 1/|x-y|.
dist = pdist2(X0, Y0);
A = 1./dist;    


%%  Strong Rank Revealing QR with given rank. 
k = 200;
f = 1.05;
[Q, R, p] = sRRQR(A, f, 'rank', k);
%disp(p(1:10));
%   approximation error 
error = norm(A(:, p) - Q * R, 'fro') / norm(A, 'fro');

%   absolute maximum of entries in inv(R11)*R12 
tmp = R(1:k, 1:k) \ R(1:k, (k+1):end);
entry = max(abs(tmp(:)));

%   print result
fprintf('Relative approx. error: %.3E\n', error);
fprintf('Maximum entry in inv(R11)*R12: %.3f\n\n', entry);


%%  Strong Rank Revealing QR with given error threshold. 
tol = 1e-4;
f = 1.01;
[Q, R, p] = sRRQR(A, f, 'tol', tol);

%   approximation error 
error = norm(A(:, p) - Q * R, 'fro') / norm(A, 'fro');

%   absolute maximum of entries in inv(R11)*R12
k = size(R, 1);
tmp = R(1:k, 1:k) \ R(1:k, (k+1):end);
entry = max(abs(tmp(:)));

%   print result
fprintf('Approximation rank: %d\n', k);
fprintf('Relative approx. error: %.3E\n', error);
fprintf('Maximum entry in inv(R11)*R12: %.3f\n\n', entry);
