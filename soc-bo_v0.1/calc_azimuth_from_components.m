function angle_deg = calc_azimuth_from_components(sxx, syy, txy)
    sxx = sxx(:); syy = syy(:); txy = txy(:);
    assert(numel(sxx)==numel(syy) && numel(sxx)==numel(txy), 'Stress component arrays must have equal lengths.');
    assert(all(isfinite([sxx;syy;txy])), 'Stress component arrays contain NaN or Inf.');

    n_points = numel(sxx);
    angle_deg = zeros(n_points,1);
    for i = 1:n_points
        horizontal_stress = [sxx(i), txy(i); txy(i), syy(i)];
        [vectors, values] = eig(0.5 * (horizontal_stress + horizontal_stress.'), 'vector');
        [~, index] = min(values);
        direction = vectors(:, index);
        angle_deg(i) = mod(rad2deg(atan2(direction(2), direction(1))), 180);
    end
end

function rmse_vec = compute_borehole_rmses(sim_mat, obs_az, bh_idx)
    unique_bhs = unique(bh_idx);
    n_bhs = numel(unique_bhs);
    fprintf('钻孔数量为： %d\n', n_bhs);
    rmse_vec = zeros(1, n_bhs);

    for b = 1:n_bhs
        idx = find(bh_idx == unique_bhs(b));
        sub_sim = sim_mat(idx, :);
        sub_obs_az = obs_az(idx);
        w = ones(length(idx), 1) / length(idx);
        rmse_vec(b) = compute_rmse_from_matrix(sub_sim, sub_obs_az, w);
    end
    fprintf('钻孔方向 RMSE：\n');
    disp(rmse_vec);
end

function rmse = compute_rmse_from_matrix(sim_mat, obs_az, weights)
    diff_sq = 0; 
    n = size(sim_mat, 1);
    for i = 1:n
        sxx = sim_mat(i, 1); 
        syy = sim_mat(i, 2); 
        txy = sim_mat(i, 4);

        [V, D] = eig([sxx txy; txy syy]); 
        [~, id] = min(diag(D)); 
        v = V(:, id);
        azi_sim = mod(rad2deg(atan2(v(2), v(1))), 180);

        u_s = [cosd(2*azi_sim); sind(2*azi_sim)]; 
        u_o = [cosd(2*obs_az(i)); sind(2*obs_az(i))];
        delta = rad2deg(acos(max(min(dot(u_s, u_o), 1), -1))) / 2;
        diff_sq = diff_sq + weights(i) * delta^2;
    end
    rmse = sqrt(diff_sq / sum(weights));
end

function total_loss = compute_TRMF_4(sim_mat, obs_mat, weights)
    sim_mat_4 = sim_mat(:, 1:4);
    obs_mat_4 = obs_mat(:, 1:4);
    total_loss = 0;
    n_points = size(sim_mat, 1);
    
    for i = 1:n_points
        v_sim = sim_mat_4(i, :);
        v_obs = obs_mat_4(i, :);
        diff_4 = v_sim - v_obs;
        misfit = norm(diff_4, 'fro') / norm(v_obs, 'fro');
        total_loss = total_loss + weights(i) * misfit;
    end
end