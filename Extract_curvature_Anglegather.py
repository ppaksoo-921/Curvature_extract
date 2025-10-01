import numpy as np
from scipy.ndimage import sobel
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import cv2
import sys
from obspy import read
from obspy.io.segy.segy import _read_segy
import time
from itertools import product
import csv
import matplotlib.font_manager as fm

SAVE_FIGURE=True
SHOW_FIGURE=False
SHOW_heatmap=True
SAVE_CSV=False
folder="low_10"

# 히트맵 타이틀 설정 (사용자가 원하는 대로 변경 가능)
HEATMAP_TITLE = f"Guideline : 10% Increased velocity model"

# Pretendard 폰트 설정
font_path = 'C:/Users/owner/Desktop/Curvature/Pretendard-Medium.otf'
try:
    # 폰트 파일을 matplotlib에 등록
    fm.fontManager.addfont(font_path)
    pretendard_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = pretendard_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    print("Pretendard 폰트 등록 성공:", pretendard_font.get_name())
except Exception as e:
    print(f"Pretendard 폰트 로드 실패: {e}")
    print("기본 폰트를 사용합니다.")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

# 전역 변수로 모든 angle gather의 곡률 데이터를 저장
all_curvature_data = []
all_position_data = []
all_gather_numbers = []


def create_curvature_visualization():
    # Function to visualize curvature data  
    # Creates a 200x210 2D array and visualizes the curvature and position data of each angle gather
    if not all_curvature_data:
        print("Saved curvature data does not exist.")
        return
    
    print("\n=== Curvature visualization start ===")
    
    INTENSITY_MIN = 0.0     
    INTENSITY_MAX = 0.007    
    SCALE_FACTOR = 4        
    
    array_height = 200 * SCALE_FACTOR
    array_width = 210 * SCALE_FACTOR
    image_array = np.full((array_height, array_width, 3), 255, dtype=np.uint8)  
    
    print(f"2D array size: {array_height}x{array_width}")
    print(f"Divide columns into 5 groups (each group: 5 columns)")
    
    # Process each angle gather data
    for group_idx, (curv_values, pos_values, gather_num) in enumerate(zip(all_curvature_data, all_position_data, all_gather_numbers)):
        if group_idx >= 42:  
            break
            
        # 해당 그룹의 열 범위 계산 (5개씩) - 해상도 배율 적용
        col_start = group_idx * 5 * SCALE_FACTOR
        col_end = col_start + 5 * SCALE_FACTOR
        
        print(f"\nGroup {group_idx+1} (Angle Gather {gather_num}):")
        print(f"  Column range: {col_start}~{col_end-1}")
        print(f"  Number of curvature data: {len(curv_values)}")
        
        # Process each position and curvature pair
        for data_idx, (pos_val, curv_val) in enumerate(zip(pos_values, curv_values)):
            if isinstance(pos_val, (int, float)) and isinstance(curv_val, (int, float)):
                # position value to row index (0~199 range)
                row_idx = int(pos_val * SCALE_FACTOR) % array_height
                
                # Calculate color based on curvature value
                if INTENSITY_MAX > INTENSITY_MIN:
                    intensity = (abs(curv_val) - INTENSITY_MIN) / (INTENSITY_MAX - INTENSITY_MIN)
                    intensity = max(0.0, min(1.0, intensity))  # 0.0~1.0 범위로 제한
                else:
                    intensity = 0.0  # 범위가 동일한 경우
                
                adjusted_intensity = intensity ** 0.5  # 제곱근을 사용하여 더 연하게
                
                # Calculate color based on negative/positive curvature
                if curv_val < 0:
                    # Negative value: green series (darker green for larger negative values)
                    # When intensity is 0: very light green (200, 255, 200)
                    # When intensity is 1: dark green (0, 100, 0)
                    red = int(200 - (200 - 0) * adjusted_intensity)      
                    green = int(255 - (255 - 100) * adjusted_intensity)  
                    blue = int(200 - (200 - 0) * adjusted_intensity)     
                else:
                    # Positive value: yellow-brown series (make yellow stronger)
                    # When intensity is 0: very light yellow (255, 255, 150)
                    # When intensity is 1: dark brown (139, 69, 19)
                    red = int(255 - (255 - 139) * adjusted_intensity)    
                    green = int(255 - (255 - 69) * adjusted_intensity)   
                    blue = int(150 - (150 - 19) * adjusted_intensity)    
                
                color = (red, green, blue)
                
                for col in range(col_start, col_end):
                    for row_offset in range(-5 * SCALE_FACTOR, 6 * SCALE_FACTOR): 
                        target_row = row_idx + row_offset
                        if 0 <= target_row < array_height and 0 <= col < array_width:
                            image_array[target_row, col] = color
                
                if data_idx < 5:  # For verification: print the first 5 data
                    print(f"  Data {data_idx+1}: position={pos_val}, curvature={curv_val:.6f}, row={row_idx}, color={color}")
    
    # Convert numpy array to PIL image
    from PIL import Image
    result_image = Image.fromarray(image_array)
    
    # Save figure
    if SAVE_FIGURE:
        heatmap_dir = f"c:/Users/owner/Desktop/Result_angle_gather/{folder}"          # Put your output directory here
        os.makedirs(heatmap_dir, exist_ok=True)
        output_path = os.path.join(heatmap_dir, f"curvature_visualization_{folder}.png")
        result_image.save(output_path, dpi=(300, 300), quality=95)
        print(f"\nCurvature visualization result saved: {output_path}")
        print(f"Image size: {result_image.size}")
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_array, aspect='auto', origin='upper')
    
    height, width = image_array.shape[:2]
    
    y_ticks = np.arange(0, 5)  # 0, 0.25, 0.5, 0.75, 1.0 
    y_labels = ['0', '0.25', '0.5', '0.75', '1.0']
    y_positions = np.linspace(0, height-1, len(y_ticks))  
    
    plt.yticks(ticks=y_positions, labels=y_labels, fontsize=24)
    plt.ylabel('Depth (km)', fontsize=24)
    
    x_ticks = np.arange(1, 42, 2)  # 1, 3, 5, 7, ..., 39, 41 
    x_positions = np.linspace(width/84, width - width/84, 42)  
    x_positions_filtered = x_positions[::2]  
    
    plt.xticks(ticks=x_positions_filtered, labels=x_ticks, fontsize=24)
    plt.xlabel('Angle Group', fontsize=24)
    
    plt.title(HEATMAP_TITLE, fontsize=28)
    
    plt.tight_layout()
    
    output_path = f"C:/Users/owner/Desktop/Result_angle_gather/guideline_{folder}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if SHOW_heatmap:
        plt.show()
    else:
        plt.close()
    
    return result_image


colors = [
    [255, 0, 0],  # Red
    [0, 255, 0],  # Green
    [0, 0, 255],  # Blue
    [255, 255, 0],  # Yellow
    [0, 255, 255],  # Cyan
    [255, 0, 255],  # Magenta
    [255, 165, 0],  # Orange
    [128, 0, 128],  # Purple
]

start_time = time.time()

def find_connected_area(grid, start):

    area=np.zeros(grid.shape)
    rows, cols = len(grid), len(grid[0])

    visited = set()             # Save visited coordinates
    connected_coords = []       # Save result
    directions = [(-1, 0), (1, 0), (0, 1)]

    # Function to explore adjacent regions: when an adjacent curve region is found, mark that region and move on to the next
    def dfs(x, y, area):

        # Stop if out of bounds, already visited, or value is not 1
        if x < 0 or x >= rows or y < 0 or y >= cols or (x, y) in visited or grid[x][y] != 255:
            return

        visited.add((x, y))
        connected_coords.append((x, y))
        area[x][y]=255

        for dx, dy in directions:
            dfs(x + dx, y + dy, area)



    # Find nearby connected curve points
    def find_curve_area():
        if not connected_coords:
            return None
        rightmost = max(connected_coords, key=lambda p: p[0])
        x, y = rightmost

        # Search for curve regions within one cell to the right and two cells down
        for dx in range(1, 2):
            for dy in range(1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and grid[nx][ny] == 255:
                    return nx, ny  # Return upon finding a curve region
        return None

    # Run the DFS function from the starting point
    dfs(start[0], start[1], area)

    while True:
        curve_start = find_curve_area()
        if not curve_start:
            break                                   # Exit if no curve region is found
        if curve_start[1]<5:
            break
            
        dfs(curve_start[0], curve_start[1], area)   # Restart DFS from the curve region

    return area

def extract_curvature(new_cp, cp_cut, where, ii):
    tot_a = []
    tot_p = []

    # Count the number of curves in the column corresponding to each starting point
    start=[]
    end=[]
    for i in range(new_cp.shape[0]-1):
        if new_cp[i][where]==0 and new_cp[i+1][where]!=0:
            start=np.append(start,i+1)
        if new_cp[i][where]!=0 and new_cp[i+1][where]==0:
            end=np.append(end,i)
    num_curve=len(start)

    line_point=np.zeros((num_curve,array.shape[1]))

    colored_image = np.zeros((cp_cut.shape[0], cp_cut.shape[1], 3), dtype=np.uint8)
    colors = [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        [255, 165, 0],  # Orange
        [128, 0, 128],  # Purple
    ]

    # Extract the maximum value per column for each curve
    check_err_case_1 = np.zeros(cp_cut.shape)  
    check_err_case_2 = np.zeros(cp_cut.shape)
    check_err_case_3 = np.zeros(cp_cut.shape)


    # Array storing the coordinates occupied by all curves
    put_coord = np.zeros((num_curve, 2000, 2))

    # Extract the maximum value per column for each curve
    for ic in range(num_curve):
        area1 = np.zeros(cp_cut.shape)

        # Store the region corresponding to the specified curve in area1
        area1 = find_connected_area(new_cp, (int(start[ic]), where))


        check_err_case_1 = check_err_case_2
        check_err_case_2 = check_err_case_3
        check_err_case_3 = area1 / 255

        # Store the phase data at the points corresponding to the curve in area2
        area2 = area1 * array

        # Identify overlapping segments with curves extracted from previous starting points (check for early overlaps)
        overlap = np.logical_and(check_err_case_3, check_err_case_2)
        coordinate = np.argwhere(overlap)
        exist = np.any(coordinate[:, 1] == 2)

        # Early connection case: i.e., a single curve where only the initial segment is separated (Abnormality 4)
        if exist:
            for i in range(3):
                max_index = np.argmax(area2[:, i])
                line_point[ic - 1][i] = max_index

            check_len = line_point[ic - 1][line_point[ic - 1] != 0]
            coor = np.argwhere(area1 != 0)

            # Store only the coordinates at positions where x-coordinate is 0, 1, or 2 in fiter
            filtered_coor = coor[np.isin(coor[:, 1], [0, 1, 2])]

            # Extract elements from filtered_coor that are not present in put_coord
            unique_in_filtered = np.array([
                row for row in filtered_coor if not (row == put_coord[ic - 1, :, :]).all(axis=1).any()
            ])

            # Insert elements from filtered_coor into previously unfilled sections
            # The key point here is using ic-1, i.e., inserting at the previous index
            zero_indices = np.where((put_coord[ic - 1, :, :] == [0, 0]).all(axis=1))[0]
            for idx, value in zip(zero_indices, unique_in_filtered):
                put_coord[ic - 1, idx, :] = value


        # Case where it cannot be considered a single curve
        else:
            for ix in range(cp_cut.shape[1]):
                max_index = np.argmax(area2[:, ix])
                line_point[ic][ix] = max_index
            check_len = line_point[ic][line_point[ic] != 0]
            coor = np.argwhere(area1 != 0)
            put_coord[ic, 0:coor.shape[0], :] = coor

    real_curve_num = 0

    # Count the actual number of curves and color them distinctly
    for ic in range(num_curve):

        # Only consider signals as valid if their length exceeds 12 grids
        check_long = np.any(put_coord[ic, :, 1] == 25)

        if check_long:
            real_curve_num = real_curve_num + 1
            for i in put_coord[ic]:
                colored_image[int(i[0]), int(i[1])] = colors[ic % 8]

    each_line = np.zeros((real_curve_num, cp_cut.shape[0], cp_cut.shape[1]))

    count_curve = 0
    # Label the data to handle only true curves more easily
    for ic in range(num_curve):
        check_long = np.any(put_coord[ic, :, 1] == 25)
        if check_long:
            count_curve = count_curve + 1
            for i in put_coord[ic]:
                each_line[count_curve - 1, int(i[0]), int(i[1])] = 1

    # Array to store the quadratic functions that best fit the data using the least squares method
    win_a = []
    win_p = []

    result_combo = np.zeros((real_curve_num, 16))
    result_error = np.zeros(real_curve_num)


    # n_array=np.copy(each_line[3])  
    if real_curve_num > 3:
        n_array=np.copy(each_line[3])
    else:
        n_array = np.zeros(cp_cut.shape)


    # Apply the least squares method across all combinations using a dictionary
    for ic in range(real_curve_num):

        # Store the curve points of each column in a dictionary
        curve_point_data = {
            'each_start_0': [],
            'each_start_1': [],
            'each_start_2': [],
            'each_start_3': [],
            'each_start_4': [],
            'each_start_5': [],
            'each_start_6': [],
            'each_start_7': [],
            'each_start_8': [],
            'each_start_9': [],
            'each_start_10': [],
            'each_start_11': [],
            'each_start_12': [],
            'each_start_13': [],
            'each_start_14': [],
            'each_start_15': []
        }
        tem_start = 0


        for j in range(new_cp.shape[1]):
            
            # For angle gathers, the data size is larger than CMP gathers,
            # so to reduce computational cost, only indices j that are multiples of 4 were processed.
            if j % 4 != 0:  
                continue
            
            key = f'each_start_{int(j/4)}'
            for i in range(new_cp.shape[0] - 1):
                if each_line[ic][i][j] != 1 and each_line[ic][i + 1][j] == 1:
                    tem_start = i + 1
                if each_line[ic][i][j] == 1 and each_line[ic][i + 1][j] != 1 and tem_start != 0:
                    tem_end = i
                    curve_point_data[key] = np.append(curve_point_data[key], (tem_start + tem_end) / 2)

        # Extract only non-empty values and generate all possible combinations (Abnormality 2, 3)
        filtered_curve_point_data = {key: values for key, values in curve_point_data.items() if len(values) > 0}
        value_lists = list(filtered_curve_point_data.values())
        combinations = list(product(*value_lists))

        # Step 4: Modify combinations that have differences >= 10
        valid_combinations = []
        for combo in combinations:
            # Skip combinations that are all zeros
            if all(x == 0 for x in combo):
                continue
                
            # Find the first index where difference >= 10
            cutoff_index = len(combo)  # Default to full length
            for i in range(len(combo) - 1):
                if abs(combo[i] - combo[i + 1]) >= 10:
                    cutoff_index = i + 1  # Cut off at the index before the large difference
                    break
            
            # Only include if the truncated combination is long enough (at least 3 elements)
            if cutoff_index >= 8:
                valid_combinations.append(combo[:cutoff_index])


        result_a = 0
        result_p = 0

        result_a_2nd = 0
        result_p_2nd = 0


        # Evaluate all combinations and select the one with the smallest error as the true curve combination
        standard_err = 1000000
        end_curve_point = 1000
        second_err = 1000


        # In angle gathers, small branch-like splits often appear due to minor errors in the angle calculation.  
        # Therefore, if separated data does not extend continuously to the end, it is considered noise and filtered out.
        if valid_combinations:
            max_length = max(len(combo) for combo in valid_combinations)
            longest_combinations = [combo for combo in valid_combinations if len(combo) == max_length]
            print(f"max_length: {max_length}, num combo: {len(longest_combinations)}")
        else:
            longest_combinations = []
        

        for combo in longest_combinations:

            y_val = np.array(combo)
            x_val = np.arange(len(y_val))

            p = y_val[0]

            X = np.vstack([(4*x_val) ** 2, np.ones(len(x_val))]).T
            adjuseted_y = y_val - p
            a = np.linalg.lstsq(X, adjuseted_y, rcond=None)[0][0]

            errors = 0
            for ix in range(len(x_val)):
                #errors = errors + abs(a * (4*ix) ** 2 - adjuseted_y[ix])
                errors = errors + abs(a * (4*ix) ** 2 - adjuseted_y[ix])

            if abs(end_curve_point - combo[len(combo)-1]) <= 30 or end_curve_point == 1000:
                if errors <= standard_err:
                    standard_err = errors
                    result_combo[ic,0:len(y_val)]=y_val
                    result_a = a
                    result_p = p
                    end_curve_point = combo[len(combo)-1]

                    print("al test")
                    print(result_combo[ic])
                    print(result_a)
                    print(errors)

            # If the deviation from the existing end_curve_point is large -> it diverges into another curve with a far offset
            # In this case, save it as a new curve
            else:
                print("separated curve detected")
                break
                if errors <= second_err:
                    second_err = errors
                    result_combo_2nd[ic, 0:len(y_val)] = y_val
                    result_a_2nd = a
                    result_p_2nd = p
                    result_error_2nd[ic] = errors
            

        win_a = np.append(win_a, result_a)
        win_p = np.append(win_p, result_p)
        result_error = np.append(result_error, standard_err)

        if result_a_2nd != 0:
            print("seuuces")
            win_a = np.append(win_a, result_a_2nd)
            win_p = np.append(win_p, result_p_2nd)
            result_error = np.append(result_error, second_err)

    tot_a.extend(win_a)
    tot_p.extend([p + ii for p in win_p])

    return colored_image, tot_a, tot_p, result_combo, result_error

# Set the detection depth for the Flood Fill algorithm
new_limit = 5000
sys.setrecursionlimit(new_limit)


# angle gather loop
for i in range(42):

    i_gather = 100 + 5*i


    # Enter the path to the file for curvature extraction
    with open(f"c:/Users/owner/Desktop/Data_angle_gather/{folder}/heatmap_{i_gather}.bin") as f:
        data = np.fromfile(f, dtype=np.float32).reshape(200,-1) 


    tot_result_a=[]
    tot_result_p=[]

    # Repeat curvature extraction using a window length of 200 grids
    win_len=200
    draw=0
    for start in range(draw,199,200):
        cmp_gather = data
        array = cmp_gather

        # Extract the curve region and store it in the 2D array `cp`
        cp=np.zeros(array.shape)

        t=0.05
        # Threshold constant -> determines from which amplitude a signal is regarded as a reflection.  
        # This value can be adjusted by the user depending on their judgment.

        scale_par = max(abs(np.max(array)), abs(np.min(array)))
        for i in range(array.shape[0]):  
            for j in range(array.shape[1]):  
                if array[i][j]>scale_par*t:
                    cp[i][j] = 255

        # Wavelength filtering: remove high-frequency noise and store the result in new_cp
        new_cp=np.zeros(cp.shape)
        status=0
        for ix in range(64):
            status = 0
            t_start = 0
            t_end = 0
            for i in range(cp.shape[0]-1):
                if cp[i][ix]==0 and cp[i+1][ix]!=0:
                    t_start=i+1
                    status=0
                if cp[i][ix]!=0 and cp[i+1][ix]==0:
                    t_end=i
                    status=1
                if status==1 and t_end-t_start>2:
                    for j in range(t_start, t_end+1):
                        new_cp[j][ix]=255

        # 1st extraction: Extract curvature using the nearest offset as the start point
        colored_image, tot_a, tot_p, result_combo, result_error = extract_curvature(new_cp, cp, 0, start)
        # 2nd extraction: Extract curvature using the second nearest offset as the start point
        colored_image_2nd, tot_a_2nd, tot_p_2nd, result_combo_2nd, result_error_2nd = extract_curvature(new_cp, cp, 1, start)


        result_a=np.copy(tot_a)
        result_p=np.copy(tot_p)

        result_colored=np.copy(colored_image)

        # Array to store the final results
        result_a_final=[]
        result_p_final=[]

        new_b=[]

        copy=np.copy(tot_a_2nd)

        # For curves detected in both 1st and 2nd extractions: find which curve in the 2nd corresponds to the i-th curve in the 1st and pair them
        pair=np.zeros(len(tot_a))
        for i , a in enumerate(tot_p):
            for j, b in enumerate(tot_p_2nd):
                if abs(a-b)<10:
                    copy[j]=0
                    if pair[i]==0:
                        pair[i]=j           # The i-th curve from the 1st extraction matches the j-th curve from the 2nd extraction
                    else:
                        print(i)

        # Plot the non-overlapping segments anew
        for i , c in enumerate(copy):
            if c!=0:
                result_a=np.append(result_a,tot_a_2nd[i])
                result_p=np.append(result_p,tot_p_2nd[i])

                result_a_final=np.append(result_a_final,tot_a_2nd[i])
                result_p_final=np.append(result_p_final,tot_p_2nd[i])


        # Compare errors in the overlapping segments: easily done using the predefined pairs
        for i , d in enumerate(pair):
            if result_error[i]<result_error_2nd[int(d)]:
                result_a_final=np.append(result_a_final,tot_a[i])
                result_p_final = np.append(result_p_final, tot_p[i])
            else:
                result_a_final=np.append(result_a_final,tot_a_2nd[int(d)])
                result_p_final = np.append(result_p_final, tot_p_2nd[int(d)])

        tot_result_a.extend(result_a_final)
        tot_result_p.extend(result_p_final)

    end_time=time.time()
    print('time: ',end_time-start_time)

    # Convert to numpy arrays if they aren't already
    tot_p = np.array(tot_result_p)
    tot_a = np.array(tot_result_a)
    # Remove elements where tot_p is 0 and corresponding tot_a elements
    non_zero_indices = tot_p != 0
    tot_p = tot_p[non_zero_indices]
    tot_a = tot_a[non_zero_indices]



    fig, ax = plt.subplots(figsize=(6, 6))
    scale_par = 10 ** (6)  

    im = ax.imshow(  
        data, origin='upper', aspect='auto', cmap='seismic',
        vmin=-scale_par, vmax=scale_par, interpolation='none'
    )
    for i in range(len(tot_a)):  
        tem_a = tot_a[i]  
        tem_p = tot_p[i]  
        x = np.arange(0, data.shape[1])  
        y = tem_a * x ** 2 + tem_p - draw  
        y = np.clip(y, 0, data.shape[0] - 1)  
        ax.plot(x, y, color='green', linewidth=3.0)
    
    ax.axis('off')  
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if SAVE_FIGURE:
        out_path = f"c:/Users/owner/Desktop/Result_angle_gather/{folder}/curvature_{i_gather}.png"  

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)  
        fig.savefig(out_path, dpi=300, bbox_inches='tight',
                    pad_inches=0, transparent=False)  
    plt.close(fig)
    
    # Save the curvature data of the current angle gather to the global list
    all_curvature_data.append(tot_a.copy())
    all_position_data.append(tot_p.copy())
    all_gather_numbers.append(i_gather)
    
    print(f"Angle Gather {i_gather}: {len(tot_a)} curvature data saved")


    ##### Code for plotting graphs #####
    if SHOW_FIGURE:
        plt.figure()
        scale_par= max(abs(np.max(array)), abs(np.min(array)))
        x_ticks = np.arange(0, 15, 3) * 3 + 11.5  
        y_ticks = np.arange(0, 1500, 50) // 5
        plt.yticks(ticks=np.arange(0, 1500, 50), labels=y_ticks)
        plt.xticks(ticks=np.arange(0, 15, 3), labels=x_ticks)

        plt.imshow(data,aspect='auto', cmap='seismic', clim=(-scale_par, scale_par), interpolation='none')
        for i in range(len(tot_a)):
            tem_a = tot_a[i]
            tem_p = tot_p[i]
            x = np.arange(0, array.shape[1])  
            y = tem_a * x**2 + tem_p - draw
            y = np.clip(y, 0, array.shape[0] - 1)
            plt.plot(x, y, color='black', linewidth=1.5)  
        plt.xlabel('offset(m)', fontsize=12)
        plt.ylabel('time (ms)', fontsize=12)
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        if SHOW_FIGURE:
            plt.show()


        plt.figure()
        scale_par= max(abs(np.max(array)), abs(np.min(array)))
        x_ticks = np.arange(0, 15, 3) * 3 + 11.5  
        y_ticks = np.arange(0, 1500, 50) // 5
        plt.yticks(ticks=np.arange(0, 1500, 50), labels=y_ticks)
        plt.xticks(ticks=np.arange(0, 15, 3), labels=x_ticks)

        plt.imshow(colored_image,aspect='auto', cmap='seismic', clim=(-scale_par, scale_par), interpolation='none')
        for i in range(len(tot_a_2nd)):
            tem_a = tot_a_2nd[i]
            tem_p = tot_p_2nd[i]
            x = np.arange(0, array.shape[1])  
            y = tem_a * x**2 + tem_p - draw
            y = np.clip(y, 0, array.shape[0] - 1)
            plt.plot(x, y, color='white', linewidth=1.5) 

       
        for x in range(result_combo.shape[0]):
            for index in range(16):
                if result_combo[x][index] != 0:  
                    plt.scatter(index*4, result_combo[x][index], color='red', s=20, marker='o')
        plt.xlabel('offset(m)', fontsize=12)
        plt.ylabel('time (ms)', fontsize=12)
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        if SHOW_FIGURE:
            plt.show()

# After all angle gather processing, create heat map
print("\n" + "="*50)
print("All Angle Gather processing is complete.")
print("="*50)

create_curvature_visualization()

print("\nCurvature analysis is complete!")