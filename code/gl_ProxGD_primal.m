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

% Group LASSO proximal gradient method
% Consider the group LASSO problem
% min 1/2*||Ax-b||_F^2 + mu*sum_i ||x(i,1:l)||_2
% where x(i,1:l) is the i-th row of matrix x, and l is the length of each group.

% This function implements the proximal gradient method for the group LASSO problem.

% 输入信息： $A$, $b$, $\mu$ ，迭代初始值 $x^0$ ，原问题对应的正则化系数 $\mu_0$ ，
% 以及提供各参数的结构体 |opts| 。

% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。

% * |out.fvec| ：每一步迭代的原始 LASSO 问题目标函数值（对应于原问题的 $\mu_0$）
% * |out.fval| ：迭代终止时的原始 LASSO 问题目标函数值（对应于原问题的 $\mu_0$）
% * |out.nrmG| ：迭代终止时的梯度范数
% * |out.tt| ：运行时间
% * |out.itr| ：迭代次数
% * |out.flag| ：记录是否收敛

function [x, iter, out] = gl_ProxGD_primal(x0, A, b, mu, opts)

% 从输入的结构体 |opts| 中读取参数或采取默认参数。

% * |opts.maxit| ：最大迭代次数
% * |opts.ftol| ：针对函数值的收敛判断条件，当相邻两次迭代函数值之差小于该值时认为该条件满足
% * |opts.gtol| ：针对梯度的收敛判断条件，当当前步梯度范数小于该值时认为该条件满足
% * |opts.alpha0| ：步长的初始值
% * |optsz.verbose| ：不为 0 时输出每步迭代信息，否则不输出
% * |opts.ls| ：标记是否线搜索
% * |opts.bb| ：标记是否采用 BB 步长
if ~isfield(opts, 'maxit'); opts.maxit = 20000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-12; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.0001; end
if ~isfield(opts, 'ls'); opts.ls = 1; end
if ~isfield(opts, 'bb'); opts.bb = 1; end

mu0 = mu;
out = struct();
k = 0;
tt = tic;
x = x0;
t = opts.alpha0;

fp = inf;
r = A*x0 - b;
g = A'*r;
tmp = .5*norm(r,'fro')^2;
tmpf = tmp + mu*sum(norms(x,2,2));
f =  tmp + mu0*sum(norms(x,2,2));
nrmG = norm(x - prox(x - g,mu),2);
out.fvec = f;

% 线搜索参数。
Cval = tmpf; Q = 1; gamma = 0.85; rhols = 1e-6;

% 迭代主循环
% 当达到最大迭代次数，或梯度或函数值的变化大于阈值时，退出迭代。
while k < opts.maxit && nrmG > opts.gtol && abs(f - fp) > opts.ftol

    % 记录上一步的迭代信息。
    gp = g;
    fp = f;
    xp = x;
    
    x = prox(xp - t * g, t * mu);

    if opts.ls
        nls = 0;
        while 1
            tmp = 0.5 * norm(A*x - b, 'fro')^2;
            tmpf = tmp + mu*sum(norms(x,2,2));
            if tmpf <= Cval - rhols*0.5*t*norm(x-xp,'fro')^2 || nls == 5
                break;
            end
            
            t = 0.2*t; nls = nls + 1;
            x = prox(xp - t * g, t * mu);
        end
        
        f = tmp + mu0*sum(norms(x,2,2));

        % 当 opts.ls=0 时，不进行线搜索。
    else
        f = 0.5 * norm(A*x - b, 'fro')^2 + mu0*sum(norms(x,2,2));
    end

    % 梯度范数的估计。
    nrmG = norm(x - xp,'fro')/t;
    r = A * x - b;
    g = A' * r;

    % 如果 |opts.bb=1| 且 |opts.ls=1| 则计算 BB 步长作为下一步迭代的初始步长。
    if opts.bb && opts.ls
        dx = x - xp;
        dg = g - gp;
        dxg = abs(dx(:)'*dg(:));
        
        if dxg > 0
            if mod(k,2) == 0
                t = norm(dx,'fro')^2/dxg;
            else
                t = dxg/norm(dg,'fro')^2;
            end
        end
        
        % 将更新得到的 BB 步长限制在阈值 [t_0,10^{12}] 内。
        t = min(max(t,opts.alpha0),1e12);
        Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + tmpf)/Q;
        
        % 如果不使用 BB 步长，则使用设定的初始步长开始下一次迭代。
    else
        t = opts.alpha0;
    end
    
    % 迭代步数加一，记录当前函数值，输出信息。
    k = k + 1;
    out.fvec = [out.fvec, f];
    if opts.verbose
        fprintf('itr: %d\tt: %e\tfval: %e\tnrmG: %e\n', k, t, f, nrmG);
    end
    
    % 特别地，除了每次迭代开始处的收敛条件外，如果连续 8 步的函数值最小值比 8 步之前的函数值超过阈值，
    % 则停止内层循环。
    if k > 8 && min(out.fvec(k-7:k)) - out.fvec(k-8) > opts.ftol
        break;
    end
end

% 当退出循环时，向外层迭代（连续化策略）报告内层迭代的退出方式，当达到最大迭代次数退出时，
% out.flag 记为 1 ，否则则为达到收敛标准，记为 0. 这个指标用于判断是否进行正则化系数的衰减。
if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

% 记录输出信息。
out.fvec = out.fvec(1:k);
out.fval = f;
iter = k;
out.tt = toc(tt);
out.nrmG = nrmG;
end

% 邻近算子
function y = prox(x, mu)
    % For each row, denote z_i = x(i,1:l), nrm = norm(z_i,2)
    % prox(x(i,1:l),mu) = (nrm - mu) * z_i / nrm, if nrm > mu
    %                   = 0, otherwise
    nr = norms(x,2,2);
    y = x .* repmat(max(nr - mu,0)./max(nr, 1e-8),1,size(x,2));
end
