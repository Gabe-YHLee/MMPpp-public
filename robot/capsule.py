import numpy as np
import torch
import scipy
import open3d as o3d

import copy

def get_points2line_dist(e1, e2, points):
    
    assert len(e1.shape) == len(e2.shape) == 1, f'e1 and e2 should be 1-dim vectors, not {e1.shape}, {e2.shape}'
    assert len(points.shape) == 2, f'Expected dimension of the input "points" is (N_points, dim), got {points.shape}'
    assert points.shape[1] == len(e1) == len(e2), f'Dimemsions of inputs are mismatch: e1: {len(e1)}, e2: {len(e2)}, points: {points.shape[1]}'
    
    N, d = points.shape

    e1 = e1.unsqueeze(0).repeat_interleave(N, dim=0)
    e2 = e2.unsqueeze(0).repeat_interleave(N, dim=0)

    if torch.norm(e1-e2, dim=1)[0] < 1e-3:
        x = e1
        dist = torch.norm(e1 - points, dim=1)
    else:
        costheta1 = torch.einsum('bi, bi -> b', (points-e1), (e2-e1)) / (torch.norm(points-e1, dim=1)*torch.norm(e2-e1, dim=1) + 1e-10)
        costheta2 = torch.einsum('bi, bi -> b', (points-e2), (e1-e2)) / (torch.norm(points-e2, dim=1)*torch.norm(e1-e2, dim=1) + 1e-10)

        dist = torch.zeros(N).to(points)

        e1_is_closest = costheta1 < 0
        e2_is_closest = costheta2 < 0
        mid_is_closest = ~e1_is_closest & ~e2_is_closest
        dist[e1_is_closest] = torch.norm(points-e1, dim=1)[e1_is_closest]
        dist[e2_is_closest] = torch.norm(points-e2, dim=1)[e2_is_closest]

        x = ((e2-e1) / torch.norm(e2-e1, dim=1, keepdim=True)) * (torch.norm(points-e1, dim=1)*costheta1).unsqueeze(1).repeat_interleave(3, dim=1) + e1
        dist[mid_is_closest] = torch.norm(x-points, dim=1)[mid_is_closest]

    return dist

def get_point2line_dist(e1, e2, p):
    
    assert len(e1.shape) == len(e2.shape) == 1, f'e1 and e2 should be 1-dim vectors, not {e1.shape}, {e2.shape}'
    assert len(p.shape) == 1, f'p should be an 1-dim vector, got {p.shape}'
    assert len(p) == len(e1) == len(e2), f'Dimemsions of inputs are mismatch: e1: {len(e1)}, e2: {len(e2)}, p: {len(p)}'

    if torch.norm(e1-e2) < 1e-3:
        x = e1
        dist = torch.norm(x - p)

    else:
        costheta1 = torch.dot((p-e1),(e2-e1)) / torch.clip(torch.norm(p-e1)*torch.norm(e2-e1), min=1e-5)
        costheta2 = torch.dot((p-e2),(e1-e2)) / torch.clip(torch.norm(p-e2)*torch.norm(e1-e2), min=1e-5)

        if costheta1 < 0:
            dist = torch.norm(p-e1)
            x = e1
        elif costheta2 < 0:
            dist = torch.norm(p-e2)
            x = e2
        else:
            x = ((e2-e1) / torch.clip(torch.norm(e2-e1), min=1.0e-5)) * (torch.norm(p-e1)*costheta1) + e1
            dist = torch.norm(x-p)

    return dist, x

def get_line2line_dist(x1, x2, y1, y2):
    
    assert len(x1.shape) == len(x2.shape) == len(y1.shape) == len(y2.shape) == 1, f'inputs should be 1-dim vectors, not x1: {x1.shape}, x2: {x2.shape}, y1: {y1.shape}, y2: {y2.shape}'

    v1 = (x2-x1) / torch.norm(x2-x1)
    v2 = (y2-y1) / torch.norm(y2-y1)

    costheta = torch.dot(v1, v2)

    if abs(costheta) < 0.9986:
        det = torch.dot(x2-x1,y2-y1) * torch.dot(y2-y1,x1-x2) - torch.dot(x2-x1,x1-x2) * torch.dot(y2-y1,y2-y1)
        t = (torch.dot(y2-y1,x1-x2) * torch.dot(x2-x1,x1-y1) - torch.dot(x2-x1,x1-x2) * torch.dot(y2-y1,x1-y1)) / det
        s = (-torch.dot(y2-y1,y2-y1) * torch.dot(x2-x1,x1-y1) + torch.dot(x2-x1,y2-y1) * torch.dot(y2-y1,x1-y1)) / det

        if (0 <= t) & (t <= 1) & (0 <= s) & (s <= 1):
            xp = (1-s)*x1 + s*x2
            yp = (1-t)*y1 + t*y2
            dist = torch.norm(xp-yp)  

            return xp, yp, dist

    dist_tmp_y1, xp_tmp1 = get_point2line_dist(x1, x2, y1)
    dist_tmp_y2, xp_tmp2 = get_point2line_dist(x1, x2, y2)

    if dist_tmp_y1 < dist_tmp_y2:
        dist_tmp = dist_tmp_y1
        xp_tmp = xp_tmp1
        yp_tmp = y1

    else:
        dist_tmp = dist_tmp_y2
        xp_tmp = xp_tmp2
        yp_tmp = y2

    dist_tmp_x1, yp_tmp1 = get_point2line_dist(y1, y2, x1)

    if dist_tmp > dist_tmp_x1:
        dist_tmp = dist_tmp_x1
        xp_tmp = x1
        yp_tmp = yp_tmp1

    dist_tmp_x2, yp_tmp2 = get_point2line_dist(y1, y2, x2)

    if dist_tmp > dist_tmp_x2:
        dist_tmp = dist_tmp_x2
        xp_tmp = x2
        yp_tmp = yp_tmp2

    return xp_tmp, yp_tmp, dist_tmp

def batch_point2line_dist(e1, e2, p):
    """_summary_

    Args:
        e1 (torch.tensor): (bs, dim)
        e2 (torch.tensor): (bs, dim)
        p (torch.tensor): (bs, dim)
    """
    bs = e1.size(0)

    dist = torch.zeros(bs).to(e1)
    idx = torch.norm(e1-e2, dim=1) >= 1e-3
    # case 1
    dist[~idx] = torch.norm(e1 - p, dim=1)[~idx]
    
    # case 2
    costheta1 = ((p-e1)*(e2-e1)).sum(1) / torch.clip(torch.norm(p-e1, dim=1)*torch.norm(e2-e1, dim=1), min=1e-5)
    costheta2 = ((p-e2)*(e1-e2)).sum(1) / torch.clip(torch.norm(p-e2, dim=1)*torch.norm(e1-e2, dim=1), min=1e-5)
    
    idx1 = idx * (costheta1 < 0)
    idx2 = idx * (costheta2 < 0)
    idx3 = idx * (costheta1 >= 0) * (costheta2 >= 0)
    
    dist[idx1] = torch.norm(p[idx1]-e1[idx1], dim=1)
    dist[idx2] = torch.norm(p[idx2]-e2[idx2], dim=1)
    
    temp = ((e2[idx3]-e1[idx3]) / torch.clip(torch.norm(
        e2[idx3]-e1[idx3], dim=1, keepdim=True), min=1.0e-5)
               ) * (torch.norm(p[idx3]-e1[idx3], dim=1)*costheta1[idx3]).view(-1, 1) + e1[idx3]
    dist[idx3] = torch.norm(temp-p[idx3], dim=1)
    return dist, torch.zeros(1)

def batch_line2line_dist(x1, x2, y1, y2):
    """_summary_

    Args:
        x1 (torch.tensor): (bs, dim)
        x2 (torch.tensor): (bs, dim)
        y1 (torch.tensor): (bs, dim)
        y2 (torch.tensor): (bs, dim)
    """
    bs = x1.size(0)
    
    v1 = (x2-x1) / torch.norm(x2-x1, dim=1, keepdim=True)
    v2 = (y2-y1) / torch.norm(y2-y1, dim=1, keepdim=True)

    costheta = (v1*v2).sum(1)

    idx1 = abs(costheta) < 0.9986
    det = ((x2-x1)*(y2-y1)).sum(1) * ((y2-y1)*(x1-x2)).sum(1) - ((x2-x1)*(x1-x2)).sum(1) * ((y2-y1)*(y2-y1)).sum(1)
    t = (((y2-y1)*(x1-x2)).sum(1) * ((x2-x1)*(x1-y1)).sum(1) - ((x2-x1)*(x1-x2)).sum(1) * ((y2-y1)*(x1-y1)).sum(1)) / torch.clip(det, min=1.0e-7)
    s = (-((y2-y1)*(y2-y1)).sum(1) * ((x2-x1)*(x1-y1)).sum(1) + ((x2-x1)*(y2-y1)).sum(1) * ((y2-y1)*(x1-y1)).sum(1)) / torch.clip(det, min=1.0e-7)
    idx2 = (0 <= t) * (t <= 1) * (0 <= s) * (s <= 1)

    xp = (1-s.view(-1,1))*x1 + s.view(-1,1)*x2
    yp = (1-t.view(-1,1))*y1 + t.view(-1,1)*y2
    
    dist = torch.zeros(bs).to(x1)
    idx = ~(idx1 * idx2)
    dist[~idx] = torch.norm(xp[~idx] - yp[~idx], dim=1)  

    dist_tmp_y1, xp_tmp1 = batch_point2line_dist(x1, x2, y1)
    dist_tmp_y2, xp_tmp2 = batch_point2line_dist(x1, x2, y2)
    dist_tmp_x1, yp_tmp1 = batch_point2line_dist(y1, y2, x1)
    dist_tmp_x2, yp_tmp2 = batch_point2line_dist(y1, y2, x2)

    dist[idx] = torch.cat([
            dist_tmp_y1[idx].view(-1,1), 
            dist_tmp_y2[idx].view(-1,1), 
            dist_tmp_x1[idx].view(-1,1), 
            dist_tmp_x2[idx].view(-1,1)], dim=1).min(dim=1).values
    
    # idx_1 = dist_tmp_y1 < dist_tmp_y2
    # dist[idx*idx_1] = dist_tmp_y1[idx*idx_1]
    # xp[idx*idx_1] = xp_tmp1[idx*idx_1]
    # yp[idx*idx_1] = y1[idx*idx_1]
    
    # dist[idx*(~idx_1)] = dist_tmp_y2[idx*(~idx_1)]
    # xp[idx*(~idx_1)] = xp_tmp2[idx*(~idx_1)]
    # yp[idx*(~idx_1)] = y2[idx*(~idx_1)]
    
    # idx_2 = dist > dist_tmp_x1
    # idx_3 = dist > dist_tmp_x2
        
    # dist[idx*(idx_2)] = dist_tmp_x1[idx*(idx_2)]
    # xp[idx*(idx_2)] = x1[idx*(idx_2)]
    # yp[idx*(idx_2)] = yp_tmp1[idx*(idx_2)]
    
    # dist[idx*(idx_3)] = dist_tmp_x2[idx*(idx_3)]
    # xp[idx*(idx_3)] = x2[idx*(idx_3)]
    # yp[idx*(idx_3)] = yp_tmp2[idx*(idx_3)]
    
    return xp, yp, dist

class Capsule(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Capsule, self).__init__()
        
        self.p1 = torch.nn.Parameter(kwargs.get('p1', torch.zeros(3)))
        self.p2 = torch.nn.Parameter(kwargs.get('p2', torch.ones(3)))
        self.r = torch.nn.Parameter(kwargs.get('r', torch.tensor(0.1)))
        self.register_parameter('p1', self.p1)
        self.register_parameter('p2', self.p2)
        self.register_parameter('r', self.r)
        
    def volume(self):
        V_sphere = 4/3 * torch.pi * self.r**3
        V_cylinder = torch.pi * self.r**2 * torch.norm(self.p1 - self.p2)
        return V_sphere + V_cylinder
    
    def distances(self, points):
        distances = get_points2line_dist(self.p1, self.p2, points) - self.r
        return distances
    
    def between(self, capsule):
        pass
    
    def get_default_mesh(self):
        
        radius = self.r.item()
        height = torch.norm(self.p1 - self.p2).item()
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius, 
            resolution=20)

        V = np.asarray(sphere.vertices)
        T = np.asarray(sphere.triangles)

        uhalf_idx = [0, *list(range(2, 402))]
        V_uhalf = V[uhalf_idx]
        V_uhalf[:, 2] += height/2
        # T_uhalf_idx = [*[2*x for x in range(41)], *list(range(81, 800))]
        # T_uhalf = T[T_uhalf_idx]

        T_uhalf = np.empty((0, 3), dtype=T.dtype)
        T_uhalf_idx = []
        for i in range(len(T)):
            if all([(x in uhalf_idx) for x in T[i]]):
                tmp_triangle = np.array([[uhalf_idx.index(T[i, 0]), uhalf_idx.index(T[i, 1]), uhalf_idx.index(T[i, 2])]], dtype=T.dtype)
                T_uhalf = np.concatenate([T_uhalf, tmp_triangle], axis=0)
                T_uhalf_idx.append(i)

        lhalf_idx = [1, *list(range(362, len(V)))]
        V_lhalf = V[lhalf_idx]
        V_lhalf[:, 2] -= height/2
        # T_lhalf_idx = [*[2*x+1 for x in range(40)], *list(range(800, 1520))]
        # T_lhalf = T[T_lhalf_idx]

        T_lhalf = np.empty((0, 3), dtype=T.dtype)
        T_lhalf_idx = []
        for i in range(len(T)):
            if all([(x in lhalf_idx) for x in T[i]]):
                tmp_triangle = np.array([[lhalf_idx.index(T[i, 0]), lhalf_idx.index(T[i, 1]), lhalf_idx.index(T[i, 2])]], dtype=T.dtype)
                T_lhalf = np.concatenate([T_lhalf, tmp_triangle], axis=0)
                T_lhalf_idx.append(i)

        V_capsule = np.concatenate((V_lhalf, V_uhalf), axis=0)
        T_capsule = np.concatenate((T_lhalf, T_uhalf + len(V_lhalf)), axis=0)

        T_cylinder = np.empty((0, 3), dtype=T_lhalf.dtype)
        T_cylinder = np.concatenate((T_cylinder, np.array([[40, 1, 40+761], [40+761, 1+761, 1]])), axis=0)
        for i in range(1, 40):
            T_cylinder = np.concatenate((T_cylinder, np.array([[i, i+1, i+761], [i+761, i+1+761, i+1]])), axis=0)

        T_capsule = np.concatenate((T_capsule, T_cylinder), axis=0)
        
        return V_capsule, T_capsule
        
    # def get_mesh(self):
    #     radius = self.r.item()
    #     height = torch.norm(self.p1 - self.p2).item()
    #     e1 = self.p1.detach().cpu().numpy()
    #     e2 = self.p2.detach().cpu().numpy()
    #     v = e2-e1
    #     w = np.cross(np.array([0, 0, 1]), v)
    #     w = w / np.linalg.norm(w)
    #     theta = np.arccos(np.dot(np.array([0, 0, 1]), v)/np.linalg.norm(v))
    #     T_SE3 = np.eye(4)
    #     T_SE3[:3, :3] = scipy.linalg.expm(skew(w*theta))
    #     T_SE3[:3, 3] = e1 + T_SE3[:3, :3] @ np.array([0, 0, height/2])
        
    #     V, T = self.get_default_mesh()
    #     V = (T_SE3 @ np.concatenate((V, np.ones((len(V), 1))), axis=1).T)[:3, :].T
        
    #     return V, T
    
# Opt Results
p1_list = [
    [-0.05465271, -0.00160026,  0.01809338],
    [-0.00020431,  0.00631866, -0.14863947],
    [-0.00611539, -0.14765818,  0.00210107],
    [ 0.07971434,  0.04523061, -0.00080541],
    [-0.08218869,  0.08021483,  0.00232922],
    [-1.5345939e-04,  8.8221747e-03, -2.3295735e-01],
    [0.08474172, 0.00705573, 0.0045368 ],
    [0.03195187, 0.03167332, 0.06886173]
]
p2_list = [
    [-3.6190036e-03,  5.1440922e-05,  5.1162861e-02],
    [ 0.00130181, -0.05217966, -0.00339115],
    [ 0.0010172,  -0.0026326,   0.05261712],
    [ 0.00334147, -0.00450623, -0.07987815],
    [-0.00809085,  0.00193277,  0.04148006],
    [-0.00389436,  0.0633805,  -0.00224684],
    [ 2.5954140e-02, -1.0050392e-03, -2.5985579e-05],
    [-0.00688292, -0.00610242,  0.07171543]
]
r_list = [
    0.10731089115142822,
    0.07845213264226913,
    0.07795113325119019,
    0.0734412893652916,
    0.07551460713148117,
    0.0707956999540329,
    0.07869726419448853,
    0.05470610037446022
]
p1s = torch.tensor(p1_list)
p2s = torch.tensor(p2_list)
rs = torch.tensor(r_list)