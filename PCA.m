function [z]=PCA
mnist=load('mnist');
[m,n]=size(mnist);
sigma=1/m*(mnist'*mnist);
[u,s,v]=svd(sigma);
u_reduce=u(1:2,:);
z=mnist*u_reduce';