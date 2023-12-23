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

% 输入信息： $A$, $b$, $\mu$ ，迭代初始值 $x^0$ 以及提供各参数的结构体 |opts| .

% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。

% * |out.fvec| ：每一步迭代的 LASSO 问题目标函数值
% * |out.fval| ：迭代终止时的 LASSO 问题目标函数值
% * |out.tt| ：运行时间
% * |out.itr| ：迭代次数
% * |out.y| ：迭代终止时的对偶变量 $y$ 的值
% * |out.nrmC| ：迭代终止时的约束违反度，在一定程度上反映收敛性
function [x, iter, out] = gl_ADMM_dual(x0, A, b, mu, opts)

% 从输入的结构体 |opts| 中读取参数或采取默认参数。
%
% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：针对函数值的停机准则，当相邻两次迭代函数值之差小于该值时认为该条件满足
% * |opts.gtol| ：针对 $x$ 的梯度的停机准则，当当前步的梯度范数小于该值时认为该条件满足
% * |opts.sigma| ：增广拉格朗日系数
% * |opts.gamma| ： $x$ 更新的步长
% * |opts.verbose| ：不为 0 时输出每步迭代信息，否则不输出
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'sigma'); opts.sigma = 0.5; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end

% 迭代准备。
tt = tic;
k = 0;
x = x0;
out = struct();

% 初始化对偶问题的对偶变量 $y$。
[m, ~] = size(A);
[~, l] = size(x0);
sm = opts.sigma;
y = zeros(m,l);

% 记录在初始时刻原问题目标函数值。
f = .5*norm(A*x - b,'fro')^2 + mu*sum(norms(x,2,2));
fp = inf;
out.fvec = f;
nrmC = inf;

% Cholesky 分解， $R$ 为上三角矩阵且 $R^\top R=A^\top A + \sigma I_m$。
% 与原始问题同样的，由于罚因子在算法的迭代过程中未变化，事先缓存 Cholesky 分解可以加速迭代过程。
W = eye(m) + sm * (A * A');
R = chol(W);

% 迭代主循环
% 迭代主循环，当 (1) 达到最大迭代次数或 (2) 目标函数的变化小于阈值或
% (3) 自变量 $x$ 的变化量小于阈值时，退出迭代循环。
while k < opts.maxit && abs(f - fp) > opts.ftol && nrmC > opts.gtol
    fp = f;

    % 对 $z$ 的更新为向无穷范数球做欧式投影
    z = proj( - A' * y + x / sm, mu);

    % 针对 $y$ 的子问题
    % 这里同样利用了事先缓存的 Cholesky 分解来加速 $y^{k+1}$ 的计算。
    h = A * (- z*sm + x) - b;
    y = R \ (R' \ h);

    % 令 $c^{k+1}=A^\top y^{k+1} + z^{k+1}$ 为等式约束的约束违反度。
    % 增广拉格朗日函数对 $x$ 的梯度为 $\frac{\partial L_\rho(y,z,x)}{\partial x}=-\sigma
    % c$。
    % 针对 $x$ 的子问题，进行一步梯度上升， $x^{k+1}=x^k - \gamma\sigma (A^\top y^{k+1} +
    % z^{k+1})$。利用 |nrmC| （约束违反度的范数）作为停机判断依据。
    c = z + A' * y;
    x = x - opts.gamma * sm * c;
    nrmC = norm(c,'fro');

    % 计算更新后的目标函数值，记录在 |out.fvec| 中。当 |opts.verbose| 不为 0 时输出详细的迭代信息。
    f = .5*norm(A*x - b,'fro')^2 + mu*sum(norms(x,2,2));
    if opts.verbose
        fprintf('itr: %4d\tfval: %e\tfeasi: %.1e\n', k, f, nrmC);
    end
    k = k + 1;
    out.fvec = [out.fvec; f];
end

% 记录输出信息。
out.y = y;
out.fval = f;
iter = k;
out.tt = toc(tt);
out.nrmC = nrmC;
end

% 辅助函数
% 到无穷范数球 $\{x\big| \|x\|_\infty \le t\}$ 的投影函数。
% For each row in x, if its norm is larger than t, then shrink it to t.
function w = proj(x, t)
    w = x;
    n = size(x, 1);
    for i = 1:n
        if norm(x(i, :)) > t
            w(i, :) = x(i, :) / norm(x(i, :)) * t;
        end
    end
end