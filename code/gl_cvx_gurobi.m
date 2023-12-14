function [x, iter, out] = gl_cvx_gurobi(x0, A, b, mu, opts)
    % Group LASSO problem using CVX with Gurobi solver
    cvx_solver gurobi

    % Extract dimensions
    [m, n] = size(A);
    [~, l] = size(b);
    
    % Define variables
    cvx_begin
        variable x(n, l)
        minimize(square_pos(norm(A * x - b, 'fro')) + mu * sum(norms(x, 2, 2)))
    cvx_end
    
    % Output results
    x = full(x);
    iter = 0; % CVX doesn't provide iteration information
    out.fval = square_pos(norm(A * x - b, 'fro')) + mu * sum(norms(x, 2, 2));
    cvx_clear;
end
