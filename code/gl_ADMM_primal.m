%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Copyright (C) 2020 Zaiwen Wen, Haoyang Liu, Jiang Hu
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 利用交替方向乘子法 (ADMM) 求解 
% 交替方向乘子法，即 the Alternating Direction Method of Multipliers

% 初始化和迭代准备
% 函数通过优化上面给出的增广拉格朗日函数，以得到 LASSO 问题的解。

% 输入信息： $A$, $b$, $\mu$ ，迭代初始值 $x^0$ 以及提供各参数的结构体 |opts| 。

% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。

% * |out.fvec| ：每一步迭代的 LASSO 问题目标函数值
% * |out.fval| ：迭代终止时的 LASSO 问题目标函数值
% * |out.tt| ：运行时间
% * |out.itr| ：迭代次数
% * |out.y| ：迭代终止时的对偶变量 $y$ 的值
% * |out.nrmC| ：约束违反度，在一定程度上反映收敛性
function [x, iter, out] = gl_ADMM_primal(x0, A, b, mu, opts)

% 从输入的结构体 |opts| 中读取参数或采取默认参数。

% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：针对函数值的停机准则，当相邻两次迭代函数值之差小于该值时认为该条件满足
% * |opts.gtol| ：针对 $y$ 的梯度的停机准则，当当前步的梯度范数小于该值时认为该条件满足
% * |opts.sigma| ：增广拉格朗日系数
% * |opts.gamma| ： $x$ 更新的步长
% * |opts.verbose| ：不为 0 时输出每步迭代信息，否则不输出
if ~isfield(opts, 'maxit'); opts.maxit = 500000; end
if ~isfield(opts, 'sigma'); opts.sigma = 1; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-14; end
if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
if ~isfield(opts, 'alpha'); opts.alpha = 0.00135; end % 0.00135
if ~isfield(opts, 'alpha2'); opts.alpha2 = 1; end

% 迭代准备。
k = 0;
tt = tic;
x = x0;
out = struct();

% 初始化 ADMM 的辅助变量 $y$, $z$，其维度均与 $x$ 相同。
[m, n] = size(A);
[~, l] = size(x0);
sm = opts.sigma;
alpha = opts.alpha;
alpha2 = opts.alpha2;
y = zeros(n,l);
z = zeros(n,l);

% 计算并记录起始点的目标函数值。
fp = inf; nrmC = inf;
f = Func(A, b, mu, x);
f0 = f;
out.fvec = f0;

% 迭代主循环
% 迭代主循环，当 (1) 达到最大迭代次数或 (2) 目标函数的变化小于阈值或 (3) 自变量 $x$ 的变化量小于阈值时，退出迭代循环。
while k < opts.maxit && abs(f - fp) > opts.ftol && nrmC > opts.gtol
    fp = f;
    
    % x^{k+1} =& x^k - \alpha \left( A^T(Ax^k - b) + y^k + \sigma (x^k - z^k) \right) \\
    % z^{k+1} =& \text{Prox}_{\alpha \mu \| \cdot \|_{1,2}} \left( z^k + \alpha \left( y^k + \sigma (x^{k+1} - z^k) \right) \right) \\
    
    % 更新 $x$ 和 $z$。
    x = x - alpha * (A' * (A * x - b) + y + sm * (x - z));
    z = prox(z + alpha2 * (y + sm * (x - z)), alpha2 * mu);

    % 以 $\|x^{k+1}-z^{k+1}\|_F$ 作为判断停机的依据。
    y = y + opts.gamma * sm * (x - z);
    f = Func(A, b, mu, x);
    nrmC = norm(x-z,'fro');
    
    % 输出每步迭代的信息。迭代步 $k$ 加一，记录当前步的函数值。
    if opts.verbose
        fprintf('itr: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
    end
    k = k + 1;
    out.fvec = [out.fvec; f];
end

% 退出循环，记录输出信息。
out.y = y;
out.fval = f;
iter = k;
out.tt = toc(tt);
c = x + y/sm;
out.nrmC = norm(c - y, inf);
end

% 辅助函数

% 邻近算子
function y = prox(x, mu)
    % For each row, denote z_i = x(i,1:l), nrm = norm(z_i,2)
    % prox(x(i,1:l),mu) = (nrm - mu) * z_i / nrm, if nrm > mu
    %                   = 0, otherwise
    nr = norms(x,2,2);
    y = x .* repmat(max(nr - mu,0)./max(nr, 1e-8),1,size(x,2));
end

% 目标函数
function f = Func(A, b, mu, x)
w = A * x - b;
f = 0.5 * norm(w, "fro")^2 + mu*sum(norms(x,2,2));
end