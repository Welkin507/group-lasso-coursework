function [x, iter, out] = gl_mosek(x0, A, b, mu, opts)
    
    [rcode, res] = mosekopt('symbcon echo(0)');

    % Extract the sizes of A and b
    [m, n] = size(A);
    [~, l] = size(b);
    
    % Define the number of variables for x, t, s, and u
    num_x = n * l;
    num_t = n;
    num_s = 1;
    num_u = 1;
    total_vars = num_x + num_t + num_s + num_u;
    
    % Objective function: 0.5 * s + mu * sum(t)
    c = [zeros(num_x, 1); mu * ones(num_t, 1); 0.5; 0];
    
    % Bounds for the variables: x is free, t >= 0, s >= 0, u is fixed
    blx = [-inf(total_vars, 1)];
    bux = [inf(total_vars, 1)];
    blx(total_vars) = 1;
    bux(total_vars) = 1;

    % Set up the problem structure
    prob = struct();
    prob.c = c;
    prob.a = zeros(0, total_vars);
    prob.b = zeros(0, 1);
    prob.blc = [];
    prob.buc = [];
    prob.blx = blx;
    prob.bux = bux;
    prob.f = [];
    prob.g = [];
    prob.accs = [];
    
    % Cone constraints for t_i >= ||x(i,:)||_2
    F = cell(n, 1);
    g = cell(n, 1);
    cones = cell(n, 1);
    for i = 1:n
        F{i} = zeros(l + 1, total_vars);
        F{i}(1, num_x + i) = 1; % t_i
        for j = 1:l
            F{i}(j + 1, (i - 1) * l + j) = 1; % x(i,j)
        end
        g{i} = zeros(l + 1, 1);
        cones{i} = [res.symbcon.MSK_DOMAIN_QUADRATIC_CONE, l + 1];
        prob.f = [prob.f; F{i}];
        prob.g = [prob.g; g{i}];
        prob.accs = [prob.accs cones{i}];
    end
    
    % Rotated cone constraint for u * s >= ||Ax - b||_F^2
    Fr = zeros(2 + m * l, total_vars);
    gr = zeros(2 + m * l, 1);
    Fr(1, num_x + num_t + 1) = 1; % s
    Fr(2, num_x + num_t + num_u + 1) = 0.5; % u/2
    for i = 1:m
        for j = 1:l
            for k = 1:n
                Fr(2 + (i-1)*l + j, (k-1)*l + j) = A(i, k); % A(i,k) * x(k,j)
            end
            gr(2 + (i-1)*l + j) = - b(i, j); % b(i,j)
        end
    end
    cone_r = [res.symbcon.MSK_DOMAIN_RQUADRATIC_CONE, 2 + m * l];
    prob.f = [prob.f; Fr];
    prob.f = sparse(prob.f);
    prob.g = [prob.g; gr];
    prob.accs = [prob.accs cone_r];
        
    % Call MOSEK
    [r, res] = mosekopt('minimize', prob);
    
    % Extract the solution
    x = res.sol.itr.xx(1:num_x);
    x = reshape(x, l, n)';
    iter = 0;
    out.fval = res.sol.itr.pobjval;
    out.res = res;
end
