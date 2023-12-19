# group-lasso-coursework
Coursework on Group LASSO problem. Optimization Methods (2023 Fall)

Consider the group LASSO problem

$$
\min _{x \in \mathbb{R}^{n \times l}} \frac{1}{2}\|A x-b\|_F^2+\mu\|x\|_{1,2}
$$

where the data $A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^{m \times l}$ and $\mu>0$ are given, and

$$
\left| x \right| _{1,2}=\sum_{i=1}^n \left| x(i, 1: l) \right|_2
$$

Note that $x(i, 1: l)$ is the $i$-th row of the matrix $x$. Here, both $x$ and $b$ are matrices but they are written in small letters for the convenience of coding. The test data are generated as follows:

```MATLAB
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
n = 512;
m = 256;
A = randn(m,n);
k = round(n*0.1); l = 2;
A = randn(m,n);
p = randperm(n); p = p(1:k);
u = zeros(n,l); u(p,:) = randn(k,l);
b = A*u;
mu = 1e-2;
```

- Solve (1.1) using CVX by calling different solvers Mosek and Gurobi.
- First write down an equivalent model of (1.1) which can be solved by calling mosek and gurobi directly, then implement the codes.
- First write down, then implement the following algorithms in Matlab (or Python) implement the codes.

