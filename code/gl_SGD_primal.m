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


% Consider the group LASSO problem
% min 1/2*||Ax-b||_F^2 + mu*sum_i ||x(i,1:l)||_2
% where x(i,1:l) is the i-th row of matrix x, and l is the length of each group.

% This function implements the subgradient descent method.

function [x, iter, out] = gl_SGD_primal(x0, A, b, mu, opts)

    % 从输入的结构体 |opts| 中读取参数或采取默认参数。

    % * |opts.maxit| ：最大迭代次数
    % * |opts.ftol| ：停机准则，当目标函数历史最优值的变化小于该值时认为满足
    % * |opts.step_type| ：步长衰减的类型（见辅助函数）
    % * |opts.alpha0| ：步长的初始值
    % * |opts.thres| ：判断小量是否被认为为 $0$ 的阈值
    if ~isfield(opts, 'maxit'); opts.maxit = 300000; end
    if ~isfield(opts, 'thres'); opts.thres = 1e-4; end
    if ~isfield(opts, 'step_type'); opts.step_type = "fixed"; end
    if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.001; end
    if ~isfield(opts, 'verbose'); opts.verbose = 0; end
    if ~isfield(opts, 'ftol'); opts.ftol = 1e-16; end

    % 初始化用于记录目标函数值、目标函数历史最优值和可微部分梯度值的矩阵。
    x = x0;
    out = struct();
    out.f_hist = zeros(1, opts.maxit);
    out.f_hist_best = zeros(1, opts.maxit);
    out.g_hist = zeros(1, opts.maxit);
    f_best = inf;

    % 迭代主循环
    % 以 |opts.maxit| 为最大迭代次数进行迭代。

    for k = 1:opts.maxit
        
        r = A * x - b;
        g = A' * r;

        % 记录可微部分的梯度的范数。
        out.g_hist(k) = norm(r, 'fro');
        
        % 记录当前目标函数值。
        f_now = 0.5 * norm(r, 'fro')^2 + mu * sum(norms(x, 2, 2));
        out.f_hist(k) = f_now;
        
        % 记录当前历史最优目标函数值。
        f_best = min(f_best, f_now);
        out.f_hist_best(k) = f_best;

        % 迭代的停止条件：当目标函数历史最优值的变化小于阈值时停止迭代。
        if k >1000 && abs(out.f_hist_best(k) - out.f_hist_best(k-999)) / abs(out.f_hist_best(1)) < opts.ftol
            break;
        end

        % For subgradient of matrix x, 
        % if one row is zero, then the subgradient of this row is zero.
        % otherwise, it's x(i,:)/||x(i,:)||_2.
        x(abs(x) < opts.thres) = 0;
        sub_g = g + mu * x ./ repmat(max(1e-8, norms(x, 2, 2)), 1, size(x, 2));

        % 利用辅助函数确定当前步步长，然后进行一步次梯度法迭代 $x^{k+1}=x^k-\alpha_k g(x^k)$。其中
        % $g(x^k)\in\partial f(x^k)$。
        alpha = set_step(k, opts);
        x = x - alpha * sub_g;
        
        % if k mod 100 == 0, print the information.
        if mod(k, 100) == 0 && opts.verbose
            fprintf('iter = %d, f = %f\n', k, f_now);
        end
    end

    % 当迭代终止时，记录当前迭代步和迭代过程。
    iter = k;
    out.fval = 0.5 * norm(A*x-b, 'fro')^2 + mu * sum(norms(x, 2, 2));
    out.f_hist = out.f_hist(1:k);
    out.f_hist_best = out.f_hist_best(1:k);
    out.g_hist = out.g_hist(1:k);
    end

    % 辅助函数
    % 函数 |set_step| 在不同的设定下决定第 $k$ 步的步长。以 $\alpha_0$ 为初始步长，步长分别为
    % 
    % * |'fixed'|： $\alpha_k=\alpha_0$
    % * |'diminishing'|： $\alpha_k=\alpha_0/\sqrt{k}$
    % * |'diminishing2'|： $\alpha_k = \alpha_0/k$

    function a = set_step(k, opts)
        type = opts.step_type;
        if strcmp(type, 'fixed')
            a = opts.alpha0;
        elseif strcmp(type, 'diminishing')
            a = opts.alpha0 / sqrt(k);
        elseif strcmp(type, 'diminishing2')
            a = opts.alpha0 / k;
        else
            error('unsupported type.');
        end
    end