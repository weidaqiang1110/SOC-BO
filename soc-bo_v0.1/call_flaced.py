import os
import time
import numpy as np
import itasca as it
from stress_direction import calc_SH_azimuth_from_stress

it.command("python-reset-state false")

# 常量定义
STEP_NUM = 2000
STRESS_OUTPUT_FILE = "stress_tensor_output.csv"
STRESS_OUTPUT_FILE_ALL = "stress_tensor_output_all.csv"
TRIGGER_FILE = "run_simulation.txt"
COMPLETED_FILE = "simulation_completed.txt"
BOUNDARY_FILE = "boundary.csv"

# 测点列表
POINTS = [
    [1200, 2250, 3000], [1200, 2250, 3100], [1200, 2250, 3200],
    [1200, 2000, 3000], [1200, 2000, 3100], [1200, 3000, 3200],
    [1800, 3000, 3000], [1800, 3000, 3100], [1800, 2500, 3200]   
]

FIELD_NAMES = ("stress-xx", "stress-yy", "stress-zz", "stress-xy", "stress-yz", "stress-xz")

while True:
    start_time = time.time()
    
    # 监听计算指令
    while not os.path.exists(TRIGGER_FILE):
        # 修正时间与报错匹配：10分钟 = 600秒
        if time.time() - start_time > 600:
            print("超时 10 分钟未收到数据，正在停止计算...")
            it.command("model interrupt")
            break 
        time.sleep(0.5)

    if not os.path.exists(TRIGGER_FILE):
        break  # 如果是因为超时跳出内层循环，则退出主程序或自行调整为 continue

    # 提高稳健性：加入 try-except 以防止 MATLAB 刚创建边界文件还未写入完成时读取崩溃
    try:
        boundary_data = np.loadtxt(BOUNDARY_FILE, delimiter=',', dtype=float)
    except Exception as e:
        print(f"读取边界文件失败，重试中: {e}")
        time.sleep(1)
        continue

    # 边界条件计算
    tidu1 = (boundary_data[3] / (4600 - 2500)) * 1e6
    sx0 = boundary_data[1] * (-1e6) - tidu1 * 4600
    
    tidu2 = (boundary_data[4] / (4600 - 2500)) * 1e6
    sy0 = boundary_data[2] * (-1e6) - tidu2 * 4600
    
    gravity_value = boundary_data[0] * (-1)
    sxy1 = boundary_data[5] * 1e6

    print("-" * 15)
    print(f"Boundary: {sx0:.2f}, {tidu1:.2f}, {sy0:.2f}, {tidu2:.2f}, {sxy1:.2f}, {sxy1:.2f}, {gravity_value:.2f}")
    print("-" * 15)

    # 运行 FLAC3D 模型 (使用 f-string 替代 format 以提高可读性)
    it.command(f"""
        model new
        model restore '..\FLAC3D Model\k22x_para1.f3sav'
        model largestrain off
               
        ; Free boundary
        zone gridpoint free velocity
        zone gridpoint initialize velocity-x 0
        zone gridpoint initialize velocity-y 0
        zone gridpoint initialize velocity-z 0
        zone face apply-remove velocity range position-x -0.1 0.1
        zone face apply-remove velocity-x range position-x -0.1 0.1
        zone face apply-remove velocity-y range position-x -0.1 0.1
        zone face apply-remove velocity range position-x 2999.0 3001
        zone face apply-remove velocity-x range position-x 2999.0 3001
        zone face apply-remove velocity-y range position-x 2999.0 3001
        zone face apply-remove velocity range position-y -0.1 0.1
        zone face apply-remove velocity-x range position-y -0.1 0.1
        zone face apply-remove velocity-y range position-y -0.1 0.1
        zone face apply-remove stress-xx range position-y -0.1 0.1
        zone face apply-remove stress-yy range position-y -0.1 0.1
        zone face apply-remove velocity range position-y 4999 5001
        zone face apply-remove velocity-x range position-y 4999 5001
        zone face apply-remove velocity-y range position-y 4999 5001
             
        ; Displacement boundary   
        zone gridpoint fix velocity-x range position-x 2999 3001 
        zone gridpoint fix velocity-y range position-y 4999.5 5000.5
        zone gridpoint fix velocity-z range position-z 2499.9 2500.1
                                    
        ; Stress boundary
        zone face apply stress-xx {sx0} gradient (0,0,{tidu1}) range position-x -0.1 0.1 position-z 2500 4660
        zone face apply stress-yy {sy0} gradient (0,0,{tidu2}) range position-y -0.1 0.1 position-z 2500 4660
        zone face apply stress-xy {sxy1} range position-x -0.1 0.1
        zone face apply stress-xy {sxy1} range position-x 2999 3001
        zone face apply stress-xy {sxy1} range position-y -0.1 0.1
        zone face apply stress-xy {sxy1} range position-y 4999.5 5000.5
               
        model gravity 0 0 {gravity_value}    
    """)

    it.fish.set("step_num", STEP_NUM)
    it.command("""
        model history mechanical ratio
        model solve
        model save 'finalCEI-it1.f3sav'
    """)

    # 提取应力数据
    stress_hist_list = []
    for x, y, z in POINTS:
        it.fish.set("local_coor_fish", (x, y, z))
        for field_name in FIELD_NAMES:
            it.fish.set("fieldname", field_name)
            it.command("""
                fish define stress_out
                    zone.field.name = fieldname
                    zone.field.method.name = "constant"
                    local_stress = zone.field.get(local_coor_fish)
                end
                [stress_out]
            """)
            stress = it.fish.get("local_stress")
            stress_hist_list.append(stress)
            
    stress_hist_array = np.array(stress_hist_list)
        
    # 保存结果
    with open(STRESS_OUTPUT_FILE, mode="w") as file:  
        file.write(','.join(map(str, stress_hist_array)) + '\n')

    with open(STRESS_OUTPUT_FILE_ALL, mode="a") as file:  
        file.write(','.join(map(str, stress_hist_array)) + '\n')

    # 计算并打印主应力方向
    angles = calc_SH_azimuth_from_stress(stress_hist_array)
    for i_ang, ang in enumerate(angles, start=1):
        print(f"Point {i_ang}: SH 与 x 轴夹角 = {ang:.2f}°")

    # 状态转换：通知 MATLAB 计算完成
    if os.path.exists(TRIGGER_FILE):
        os.rename(TRIGGER_FILE, COMPLETED_FILE)