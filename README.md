# group-lasso-coursework
Coursework on Group LASSO problem. Optimization Methods (2023 Fall)

Consider the group LASSO problem
    $$
    \min _{x \in \mathbb{R}^{n \times l}} \frac{1}{2}\|A x-b\|_F^2+\mu\|x\|_{1,2}
    $$

where the data $A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^{m \times l}$ and $\mu>0$ are given, and
    $$
    \|x\|_{1,2}=\sum_{i=1}^n\|x(i, 1: l)\|_2 .
    $$

    Note that $x(i, 1: l)$ is the $i$-th row of the matrix $x$. Here, both $x$ and $b$ are matrices but they are written in small letters for the convenience of coding. The test data are generated as follows:

'''MATLAB
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
'''

- Solve (1.1) using CVX by calling different solvers Mosek and Gurobi.
- First write down an equivalent model of (1.1) which can be solved by calling mosek and gurobi directly, then implement the codes.
- First write down, then implement the following algorithms in Matlab (or Python) implement the codes.

Here is the result.
    \begin{table}[H]
        \centering
        \resizebox{\textwidth}{!}{
        \begin{tabular}{lcccccccc}
        \hline
        Method & CPU (s) & Iterations & Optimal Value & Sparsity & Err. to Exact & Err. to CVX-Mosek & Err. to CVX-Gurobi \\
        \hline
        CVX-Mosek & 2.66 & 11 & $5.80563 \times 10^{-1}$ & 0.103 & $1.90 \times 10^{-5}$ & $0.00 \times 10^{0}$ & $4.74 \times 10^{-7}$ \\
        CVX-Gurobi & 0.72 & 25 & $5.80563 \times 10^{-1}$ & 0.103 & $1.87 \times 10^{-5}$ & $4.74 \times 10^{-7}$ & $0.00 \times 10^{0}$ \\
        Mosek & 1.29 & 10 & $5.80556 \times 10^{-1}$ & 0.103 & $3.75 \times 10^{-5}$ & $1.85 \times 10^{-5}$ & $1.88 \times 10^{-5}$ \\
        Gurobi & 0.31 & 12 & $5.80556 \times 10^{-1}$ & 0.105 & $3.77 \times 10^{-5}$ & $1.87 \times 10^{-5}$ & $1.90 \times 10^{-5}$ \\
        SGD Primal & 7.26 & 242805 & $5.83301 \times 10^{-1}$ & 0.966 & $2.12 \times 10^{-4}$ & $2.06 \times 10^{-4}$ & $2.06 \times 10^{-4}$ \\
        ProxGD Primal & 0.87 & 18465 & $5.80556 \times 10^{-1}$ & 0.103 & $3.74 \times 10^{-5}$ & $1.84 \times 10^{-5}$ & $1.88 \times 10^{-5}$ \\
        FProxGD Primal & 0.29 & 6628 & $5.80556 \times 10^{-1}$ & 0.103 & $3.71 \times 10^{-5}$ & $1.81 \times 10^{-5}$ & $1.85 \times 10^{-5}$ \\
        ALM Dual & 4.71 & 1423 & $5.81258 \times 10^{-1}$ & 0.987 & $5.31 \times 10^{-4}$ & $5.37 \times 10^{-4}$ & $5.36 \times 10^{-4}$ \\
        ADMM Dual & 0.19 & 320 & $5.80633 \times 10^{-1}$ & 0.848 & $6.66 \times 10^{-5}$ & $5.27 \times 10^{-5}$ & $5.29 \times 10^{-5}$ \\
        ADMM Primal & 23.35 & 174084 & $5.80635 \times 10^{-1}$ & 0.872 & $1.06 \times 10^{-4}$ & $9.32 \times 10^{-5}$ & $9.34 \times 10^{-5}$ \\
        \hline
        \end{tabular}
        }
        \caption{Comparison of optimization methods}
    \end{table}
