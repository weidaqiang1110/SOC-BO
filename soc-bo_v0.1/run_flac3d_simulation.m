function [stress_mat, azimuths] = run_flac3d_simulation(x, num_points)
    % 返回应力矩阵（每行一个测点, 6个应力分量）和最大主应力方向 
    
    writematrix(x, 'boundary.csv', 'Delimiter', ',');
    
    % 提高稳健性：使用 MATLAB 内置函数创建空文件，替代系统依赖的命令
    fid = fopen('run_simulation.txt', 'w');
    if fid ~= -1
        fclose(fid);
    end
    
    fprintf('Waiting for FLAC3D simulation to complete ...\n');
    t0 = tic;
    
    while ~exist('simulation_completed.txt', 'file')
        pause(1);
        % 修正注释与逻辑的统一 (3600秒 = 1小时)
        if toc(t0) > 3600
            error('FLAC3D simulation timeout (1 hour).');
        end
    end
    fprintf('The FLAC3D simulation results have been read.\n');
    
    % 使用 readmatrix 替代不推荐的 csvread
    stress_data = readmatrix("stress_tensor_output.csv");
    stress_mat = reshape(stress_data', [], num_points)';
    
    % 提取前4个应力分量并计算方位角（删除了原代码无意义的自我赋值）
    azimuths = calc_azimuth_from_components(stress_mat(:,1), stress_mat(:,2), stress_mat(:,4));
    
    delete("simulation_completed.txt");
end