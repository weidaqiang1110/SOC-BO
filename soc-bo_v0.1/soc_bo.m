%% PIBO_MultiConstraint_Boreholes_NA.m
% 程序思路:
% 1. 读取带噪声观测值
% 2. 读取前20次X和应力响应，计算目标和约束
% 3. 多约束贝叶斯优化主循环

clear; clc; close all;
rng(1);

% 提高稳健性：检查文件是否存在后再删除，避免报错
if exist('run_simulation.txt', 'file'), delete('run_simulation.txt'); end
if exist('simulation_completed.txt', 'file'), delete('simulation_completed.txt'); end 

%% ==========================================================
% 1. 问题定义与数据分组
% ===========================================================
fprintf('>>> [Step 1] 初始化任务...\n');

n_boreholes = 3;     % 钻孔数量
points_per_bh = 3;   % 每个钻孔测点数量
total_points = n_boreholes * points_per_bh;  % 总测量数量

% 规范化命名
obs_stress_data_tbl = readtable('hf_observation_noise.xlsx');
obs_stress_data_pa = obs_stress_data_tbl .* (-1e6);
obs_stress_mat = table2array(obs_stress_data_pa(:, {'sxx', 'syy', 'szz', 'sxy', 'syz', 'sxz'}));

disp('Observed Stress (Pa): ');
disp(obs_stress_mat);

% 实测主应力方向
obs_azimuths = readmatrix('SH Azimuth with noise.xlsx');
disp('Observed Azimuths: ');
disp(obs_azimuths');   % 输出一行

% 权重分配 (用 ones 替换冗长的手写数组)
weights = ones(1, 12);
fprintf('权重维度为： %d\n', length(weights));
weights = weights / sum(weights);   % 归一化

% 封装观察数据结构体
obs_data.stress_mat = obs_stress_mat;
obs_data.azimuths = obs_azimuths;
obs_data.weights = weights;

% 定义每个测点属于哪个钻孔
obs_data.bh_idx = reshape(repmat(1:n_boreholes, points_per_bh, 1), [], 1);
obs_data.n_boreholes = n_boreholes;

fprintf('钻孔数量： %d | 总测点数： %d\n', n_boreholes, total_points);

%% ==========================================================
% 2. 算法配置
% ===========================================================
n_dims = 6;   
cfg.n_init = 20;
cfg.n_iter = 80;
cfg.bounds = [
     9.4,  7.0,  9.0, 11.0, 11.0, -4.0;
    10.2, 14.5, 16.5, 17.5, 17.5,  4.0
];

% 约束算法参数
cfg.tol_init_guess = [15.0, 15.0, 15.0];
cfg.tol_final = [12.0, 12.0, 12.0];
cfg.decay_rate = 5.0;

% 退火策略参数
cfg.beta_start = 0.8;
cfg.anneal_slope = 10;

%% ==========================================================
% 3. 初始采样
% ===========================================================
fprintf('\n>>> [Step 2] 初始采样...\n');
X = lhsdesign(cfg.n_init, n_dims) .* (cfg.bounds(2,:) - cfg.bounds(1,:)) + cfg.bounds(1,:);
writematrix(X, 'X_20.xlsx');

Y_TRMF = zeros(cfg.n_init, 1);
Y_RMSE_BH = zeros(cfg.n_init, n_boreholes);

for i = 1 : cfg.n_init
    fprintf('\n>>> 第 %d 次初始采样\n', i);
    disp(X(i,:));
    
    [sim_mat, sim_azi] = run_flac3d_simulation(X(i,:), total_points);
    
    % 添加了分号阻止刷屏，同时修复了拼写 (contrast)
    contrast_mat = [obs_data.stress_mat(:,1) sim_mat(:,1) ...
                    obs_data.stress_mat(:,2) sim_mat(:,2) ...
                    obs_data.stress_mat(:,3) sim_mat(:,3) ...
                    obs_data.stress_mat(:,4) sim_mat(:,4) ...
                    obs_data.stress_mat(:,5) sim_mat(:,5) ...
                    obs_data.stress_mat(:,6) sim_mat(:,6)];

    % 计算全局 TRMF
    Y_TRMF(i) = compute_TRMF_4(sim_mat, obs_data.stress_mat, obs_data.weights);
    
    % 计算每个钻孔的独立 RMSE
    rmse_vec = compute_borehole_rmses(sim_mat, obs_data.azimuths, obs_data.bh_idx);
    Y_RMSE_BH(i,:) = rmse_vec;

    rmse_ini_str = sprintf('%.1f ', rmse_vec);
    fprintf('Iter %2d | Loss: %.4f | BH_RMSEs: [%s]\n', i, Y_TRMF(i), rmse_ini_str);
end

% cfg.tol_init_adaptive = median(Y_RMSE_BH, 1);
cfg.tol_init = cfg.tol_init_guess;
fprintf('Init Tol: [%.1f, %.1f, %.1f]\n', cfg.tol_init);

% 修正逻辑：补充后续迭代次数 n_iter 预分配空间，而非 n_init
history.loss = [Y_TRMF; zeros(cfg.n_iter, 1)];
history.rmse_bh = [Y_RMSE_BH; zeros(cfg.n_iter, n_boreholes)];
history.tol_matrix = zeros(cfg.n_init + cfg.n_iter, n_boreholes);    

%% ==========================================================
% 4. 多约束贝叶斯优化主循环
% ===========================================================
fprintf('\n>>> [Step 3] 进入多约束PIBO主循环...\n');

for k = 1:cfg.n_iter
    idx = cfg.n_init + k;
    progress = (k - 1) / cfg.n_iter;

    % 更新参数策略
    curr_tol = cfg.tol_final + (cfg.tol_init - cfg.tol_final) * exp(-cfg.decay_rate * progress);
    history.tol_matrix(idx, :) = curr_tol;
    
    sigmoid_val = 1 / (1 + exp(-cfg.anneal_slope * (progress - 0.5)));
    curr_beta = cfg.beta_start + (1.0 - cfg.beta_start) * sigmoid_val;

    % 训练GP群（1个GP目标+M个约束GP）
    gp_obj = fitrgp(X, Y_TRMF, 'KernelFunction', 'matern52', 'Standardize', true);

    % 约束GPs （用Cell数组存储）
    gp_cons = cell(n_boreholes, 1);
    for b = 1:n_boreholes
        gp_cons{b} = fitrgp(X, Y_RMSE_BH(:, b), 'KernelFunction', 'matern52',...
            'Standardize', true, 'SigmaLowerBound', 1e-3);
    end

    % 采集函数优化
    x_new = optimization_acq_multi_constraint(gp_obj, gp_cons, curr_tol, curr_beta,...
        cfg.bounds, min(Y_TRMF));
    disp(x_new);

    % 模拟与评估
    [sim_mat_new, sim_azi] = run_flac3d_simulation(x_new, total_points);
    
    contrast_mat = [obs_data.stress_mat(:,1) sim_mat_new(:,1) ...
                    obs_data.stress_mat(:,2) sim_mat_new(:,2) ...
                    obs_data.stress_mat(:,3) sim_mat_new(:,3) ...
                    obs_data.stress_mat(:,4) sim_mat_new(:,4) ...
                    obs_data.stress_mat(:,5) sim_mat_new(:,5) ...
                    obs_data.stress_mat(:,6) sim_mat_new(:,6)];

    loss_new = compute_TRMF_4(sim_mat_new, obs_data.stress_mat, obs_data.weights);
    rmse_vec_new = compute_borehole_rmses(sim_mat_new, obs_data.azimuths, obs_data.bh_idx);

    % 更新记录
    X = [X; x_new];
    Y_TRMF = [Y_TRMF; loss_new];
    Y_RMSE_BH = [Y_RMSE_BH; rmse_vec_new];
    
    history.loss(idx) = loss_new;
    history.rmse_bh(idx, :) = rmse_vec_new;

    rmse_str = sprintf('%.1f ', rmse_vec_new);
    curr_tol_str = sprintf('%.1f ', curr_tol);
    x_str = sprintf('%.2f ', x_new);
    fprintf('Iter %2d | Tol: [%s] | Loss: %.4f | BH_RMSEs: [%s] | X: [%s]\n', ...
        k, curr_tol_str, loss_new, rmse_str, x_str);
end

%% ==========================================================
% 5. 结果保存
% ===========================================================
% 统一替换 csvwrite 为推荐的 writematrix
writematrix(X, 'X_100.csv');
writematrix(Y_RMSE_BH, 'Y_RMSE_BH_100.csv');
writematrix(Y_TRMF, 'Y_TRMF_100.csv');
save('history.mat', 'history');

write_data = [Y_TRMF, Y_RMSE_BH, X];
writematrix(write_data, 'results.csv');

% 写最优边界
[~, min_row_idx] = min(write_data(:,1));
extracted_bc = write_data(min_row_idx, end-size(cfg.bounds, 2)+1 : end);
writematrix(extracted_bc, 'boundary.csv');