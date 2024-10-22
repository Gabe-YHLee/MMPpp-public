import numpy as np

def add_components(f, name, x, y, z, size_x, size_y, size_z):
    f.write(f'  <link name="{name}">' + '\n')
    f.write('    <contact>' + '\n')
    f.write('        <lateral_friction value="1"/>' + '\n')
    f.write('    </contact>' + '\n')
    f.write('\n')
    f.write('    <inertial>' + '\n')
    f.write('      <origin rpy="0 0 0" xyz="0 0 0"/>' + '\n')
    f.write('      <mass value=".0"/>' + '\n')
    f.write('      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>' + '\n')
    f.write('    </inertial>' + '\n')
    f.write('\n')
    f.write('    <visual>' + '\n')
    f.write(f'      <origin rpy="0 0 0" xyz="{x} {y} {z}"/>' + '\n')
    f.write('      <geometry>' + '\n')
    f.write(f'        <box size="{size_x} {size_y} {size_z}"/>' + '\n')
    f.write('      </geometry>' + '\n')
    f.write('      <material name="table_gray"/>' + '\n')
    f.write('    </visual>' + '\n')
    f.write('\n')
    f.write('    <collision>' + '\n')
    f.write(f'      <origin rpy="0 0 0" xyz="{x} {y} {z}"/>' + '\n')
    f.write('      <geometry>' + '\n')
    f.write(f'        <box size="{size_x} {size_y} {size_z}"/>' + '\n')
    f.write('      </geometry>' + '\n')
    f.write('    </collision>' + '\n')
    f.write('\n')
    f.write('  </link>' + '\n')
    f.write('\n')

def add_joint(f, name, parent_name, child_name):
    f.write(f'  <joint name="{name}" type="fixed">' + '\n')
    f.write(f'    <parent link="{parent_name}"/>' + '\n')
    f.write(f'    <child link="{child_name}"/>' + '\n')
    f.write('    <origin rpy="0 0 0.0" xyz="0 0 0.0"/>' + '\n')
    f.write('  </joint>' + '\n')
    f.write('\n')
    
def random_sample_positions(n, l, min_interval):
    if not n == 1:
        done = False
        while not done:
            output = np.random.rand(n-1) * l
            temp = np.abs(output.reshape(1, n-1) - output.reshape(n-1, 1))
            if n == 2:
                if output.min() > min_interval:
                    if output.max() < l - min_interval:
                        done = True
            else:
                if temp[temp > 0].min() > min_interval:
                    if output.min() > min_interval:
                        if output.max() < l - min_interval:
                            done = True
        return output
    else:
        return np.array([])

def shelf_sampler():
    done = False
    while not done:
        l_x = 0.4 + 0.1*np.random.rand()
        l_y = 1.0 + 0.2*np.random.rand()
        l_z = 1.0 + 0.2*np.random.rand()
        thickness = 0.02 + 0.005*np.random.rand()
        n_y = 1 + np.random.randint(4)
        n_z = 1 + np.random.randint(4)
        if n_y + n_z == 1:
            done = False
        else: 
            done = True
            min_interval = 0.1
            ys = random_sample_positions(n_y, l_y, min_interval)
            ys = ys - l_y/2
            zs = random_sample_positions(n_z, l_z, min_interval)
    return l_x, l_y, l_z, ys, zs, thickness

def box_urdf_writer(path, size_x, size_y, size_z):
    f = open(path, 'w')
    f.write('<?xml version="1.1" ?> \n')
    f.write('<robot name="plane"> \n')
    f.write('\n')
    add_components(f, 'box', 0.0, 0.0, 0.0, size_x, size_y, size_z)
    
    f.write('  <material name="table_gray">' + '\n')
    f.write('    <color rgba=".7686 .6431 .5176 1."/>' + '\n')
    f.write('  </material>' + '\n')
    f.write('</robot>' + '\n')
    f.close()

def shelf_urdf_writer(path, l_x, l_y, l_z, ys, zs, thickness):
    f = open(path, 'w')
    f.write('<?xml version="1.1" ?> \n')
    f.write('<robot name="plane"> \n')
    f.write('\n')

    add_components(f, 'shelve_bottom', 0.0, 0.0, thickness/2, l_x, l_y, thickness)
    add_components(f, 'shelve_top', 0.0, 0.0, l_z - thickness/2, l_x, l_y, thickness)
    add_joint(f, 'joint_bt', 'shelve_bottom', 'shelve_top')   
    add_components(f, 'shelve_back', -l_x/2 + thickness/2, 0, l_z/2, thickness, l_y, l_z - 2*thickness)
    add_joint(f, 'joint2_bb', 'shelve_bottom', 'shelve_back')  
    add_components(f, 'shelve_left', thickness/2, - l_y/2 + thickness/2, l_z/2, l_x - thickness, thickness, l_z - 2*thickness)
    add_joint(f, 'joint_bl', 'shelve_bottom', 'shelve_left')  
    add_components(f, 'shelve_right', thickness/2, l_y/2 - thickness/2, l_z/2, l_x - thickness, thickness, l_z - 2*thickness)
    add_joint(f, 'joint_br', 'shelve_bottom', 'shelve_right')  

    for iy, y_ in enumerate(ys):
        add_components(f, f'shelve_vertical_{iy}', thickness/2, y_ - y_/np.abs(y_)*thickness/2, l_z/2, l_x - thickness, thickness, l_z - 2*thickness)
        add_joint(f, f'joint_bv{iy}', 'shelve_bottom', f'shelve_vertical_{iy}')  

    for iz, z_ in enumerate(zs):
        add_components(f, f'shelve_horizontal_{iz}', thickness/2, 0, z_ - z_/np.abs(z_)*thickness/2, l_x - thickness, l_y - 2*thickness, thickness)
        add_joint(f, f'joint_hv{iz}', 'shelve_bottom', f'shelve_horizontal_{iz}')  
        
    f.write('  <material name="table_gray">' + '\n')
    f.write('    <color rgba=".7686 .6431 .5176 1."/>' + '\n')
    f.write('  </material>' + '\n')
    f.write('</robot>' + '\n')
    f.close()
    
def suface_points_sampler_box(num_points, x, y, z, lx, ly, lz):
    Axy = lx*ly
    Azx = lx*lz
    Ayz = ly*lz
    
    np_xy = int(num_points * (Axy)/(Axy + Azx + Ayz)/2)
    np_yz = int(num_points * (Ayz)/(Axy + Azx + Ayz)/2)
    np_zx = int(num_points * (Azx)/(Axy + Azx + Ayz)/2)
    
    pcs = []
    pcs.append(np.hstack([
        -lx/2 + lx*np.random.rand(np_xy, 1),
        -ly/2 + ly*np.random.rand(np_xy, 1),
        lz/2 * np.ones((np_xy, 1))]))
    
    pcs.append(np.hstack([
        -lx/2 + lx*np.random.rand(np_xy, 1),
        -ly/2 + ly*np.random.rand(np_xy, 1),
        -lz/2 * np.ones((np_xy, 1))]))
    
    pcs.append(np.hstack([
        lx/2 * np.ones((np_yz, 1)),
        -ly/2 + ly*np.random.rand(np_yz, 1),
        -lz/2 + lz*np.random.rand(np_yz, 1)]))
    
    pcs.append(np.hstack([
        -lx/2 * np.ones((np_yz, 1)),
        -ly/2 + ly*np.random.rand(np_yz, 1),
        -lz/2 + lz*np.random.rand(np_yz, 1)]))
   
    pcs.append(np.hstack([
        -lx/2 + lx*np.random.rand(np_zx, 1),
        -ly/2 * np.ones((np_zx, 1)),
        -lz/2 + lz*np.random.rand(np_zx, 1)]))
    
    pcs.append(np.hstack([
        -lx/2 + lx*np.random.rand(num_points-2*np_xy-2*np_yz-np_zx, 1),
        ly/2 * np.ones((num_points-2*np_xy-2*np_yz-np_zx, 1)),
        -lz/2 + lz*np.random.rand(num_points-2*np_xy-2*np_yz-np_zx, 1)]))
    
    pcs = np.vstack(pcs)
    pcs = pcs + np.array([[x, y, z]])
    return pcs

def shelf_surface_points_sampler(num_points, l_x, l_y, l_z, ys, zs, thickness):
    Axy = l_x*l_y
    Azx = l_z*l_x
    Ayz = l_y*l_z
    
    total_area =  Ayz + (2 + len(ys))*Azx + (2 + len(zs))*Axy 
    
    nxy = int(num_points * Axy/total_area)
    nzx = int(num_points * Azx/total_area) 
    nyz = num_points - (2+len(ys))*nzx - (2+len(zs))*nxy
    
    pcs = []
    pcs.append(suface_points_sampler_box(nxy, 0.0, 0.0, thickness/2, l_x, l_y, thickness))
    pcs.append(suface_points_sampler_box(nxy, 0.0, 0.0, l_z - thickness/2, l_x, l_y, thickness))
    pcs.append(suface_points_sampler_box(nyz, -l_x/2 + thickness/2, 0, l_z/2, thickness, l_y, l_z - 2*thickness))
    pcs.append(suface_points_sampler_box(nzx, thickness/2, - l_y/2 + thickness/2, l_z/2, l_x - thickness, thickness, l_z - 2*thickness))
    pcs.append(suface_points_sampler_box(nzx, thickness/2, l_y/2 - thickness/2, l_z/2, l_x - thickness, thickness, l_z - 2*thickness))
    for y_ in ys:
        pcs.append(suface_points_sampler_box(nzx, thickness/2, y_ - y_/np.abs(y_)*thickness/2, l_z/2, l_x - thickness, thickness, l_z - 2*thickness))
    for z_ in zs:
        pcs.append(suface_points_sampler_box(nxy, thickness/2, 0, z_ - z_/np.abs(z_)*thickness/2, l_x - thickness, l_y - 2*thickness, thickness))
    return np.concatenate(pcs)

def shelf_candidate_3dgrid_points(l_x, l_y, l_z, ys, zs, thickness):
    margin = 0.05
    list_candpcs = []
    ys.sort()
    ys = np.hstack([np.array([-l_y/2]), ys, np.array([l_y/2])])
    zs.sort()
    zs = np.hstack([np.array([0]), zs, np.array([l_z])])
    for iy, fy in zip(ys[:-1], ys[1:]):
        for iz, fz in zip(zs[:-1], zs[1:]):
            xx, yy, zz = np.meshgrid(
                np.linspace(-l_x/2 + margin, l_x/2 + margin, 5),
                np.linspace(iy + margin, fy - margin, 5),
                np.linspace(iz + margin, fz - margin, 5),                
            )
            list_candpcs.append(
                np.hstack([
                xx.flatten().reshape(-1, 1),
                yy.flatten().reshape(-1, 1),
                zz.flatten().reshape(-1, 1)])
            )
    return list_candpcs

def sample_two_points(list_candpcs):
    done = False
    while not done:
        i = np.random.randint(len(list_candpcs))
        j = np.random.randint(len(list_candpcs))
        if i!=j:
            done = True
    ii = np.random.randint(len(list_candpcs[i]))
    jj = np.random.randint(len(list_candpcs[j]))
    return list_candpcs[i][ii], list_candpcs[j][jj]

def sample_random_gripper_orientations(shelf_rot, z_margin=np.pi/6, y_margin=np.pi/6):
        base_rot = shelf_rot@np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        theta_z = -z_margin + np.random.rand()*2*z_margin
        theta_y = -y_margin + np.random.rand()*2*y_margin
        base_rot = base_rot@np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        base_rot = base_rot@np.array([[np.cos(theta_y), 0, -np.sin(theta_y)], [0, 1, 0], [np.sin(theta_y), 0, np.cos(theta_y)]])
        return base_rot