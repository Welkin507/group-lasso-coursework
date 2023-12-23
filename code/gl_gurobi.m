function [x, iter, out] = gl_gurobi(x0, A, b, mu, opts)
    % Get the size of A and b
    [m, n] = size(A);
    [~, l] = size(b);

    % Initialize Gurobi model
    model.modelsense = 'min';

    % Variables [vec(x); vec(y); t]
    num_vars = n*l + m*l + n;

    % Objective function: c' * x + 0.5 * x' * Q * x
    model.obj = [zeros(n*l, 1); zeros(m*l, 1); mu * ones(n, 1)];
    diag1 = zeros(n*l, 1);
    diag2 = ones(m*l, 1) / 2; 
    diag3 = zeros(n, 1);
    diagonals = [diag1; diag2; diag3];
    model.Q = spdiags(diagonals, 0, num_vars, num_vars);

    % Constraints: Ax - y = b
    Aeq = [kron(eye(l), A), -eye(m*l), zeros(m*l, n)];
    model.A = sparse(Aeq);
    model.rhs = b(:);
    model.sense = repmat('=', m*l, 1);

    % Bounds on variables
    model.lb = [-inf(num_vars - n, 1); zeros(n, 1)];
    model.ub = inf(num_vars, 1);

    % Quadratic constraints: t_i^2 >= ||x(i,:)||_2^2
    
    for i = 1:n
        model.quadcon(i).Qrow = [i:n:n*l, num_vars - n + i];
        model.quadcon(i).Qcol = [i:n:n*l, num_vars - n + i];
        model.quadcon(i).Qval = [ones(1, l), -1];
        model.quadcon(i).q = zeros(num_vars, 1);
        model.quadcon(i).rhs = 0;
        model.quadcon(i).sense = '<';
    end

    % Gurobi parameters
    params = struct();
    if exist('opts', 'var')
        if isfield(opts, 'TimeLimit')
            params.TimeLimit = opts.TimeLimit;
        end
    end

    % Solve the problem
    result = gurobi(model, params);

    % Extract the solution
    x = reshape(result.x(1:n*l), n, l);
    iter = result.itercount;
    out.fval = result.objval;
    out.status = result.status;
end
