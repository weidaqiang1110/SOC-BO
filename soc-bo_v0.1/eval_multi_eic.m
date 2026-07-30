function vals = eval_multi_eic(X, gp_obj, gp_cons, tol, beta, y_best)
    [mu_o, s_o] = predict(gp_obj, X);
    s_o = max(s_o, 1e-12);

    z = (y_best - mu_o) ./ s_o;
    ei = (y_best - mu_o) .* normcdf(z) + s_o .* normpdf(z);
    ei = max(ei, 0);
    
    if beta == 0
        vals = ei;
        vals(~isfinite(vals)) = 0;
        return
    end
        
    n_samples = size(X, 1);
    log_total_pof = zeros(n_samples, 1);
    n_cons = length(gp_cons);
    
    for b = 1:n_cons
        [mu_c, s_c] = predict(gp_cons{b}, X);
        s_c = max(s_c, 1e-12);
        
        pof_b = normcdf((tol(b) - mu_c) ./ s_c);
        pof_b = min(max(pof_b, 1e-12), 1 - 1e-12);
        
        log_total_pof = log_total_pof + log(pof_b);
    end
    
    log_total_pof = log_total_pof / 3; 
    vals = ei .* exp(log_total_pof * beta);
    vals(isnan(vals)) = 0;
end

function x_best = optimization_acq_multi_constraint(gp_obj, gp_cons, tol, beta, bounds, y_best)
    n_vars = size(bounds, 2);
    fitness_fcn = @(x) -1 * eval_multi_eic(x, gp_obj, gp_cons, tol, beta, y_best);

    % GA 粗寻优 (清理了重复注释的冗余配置)
    opts_ga = optimoptions('ga', 'Display', 'off', 'PopulationSize', 100, 'MaxGenerations', 50);
    [x_ga, ~] = ga(fitness_fcn, n_vars, [], [], [], [], bounds(1,:), bounds(2,:), [], opts_ga);

    % SQP 精修 (去除了无效的 try-catch，因为设置容差后报错率极低)
    opts_fmincon = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', ...
        'MaxFunctionEvaluations', 2000, 'OptimalityTolerance', 1e-8);
        
    x_best = fmincon(fitness_fcn, x_ga, [], [], [], [], bounds(1,:), bounds(2,:), [], opts_fmincon);
end