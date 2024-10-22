import numpy as np
import open3d as o3d

class Superquadric():
    
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], resolution=30):
        
        self.mesh = mesh_superquadric(parameters, resolution=resolution)
        self.mesh.paint_uniform_color(color)
        self.mesh.transform(SE3)

def mesh_superquadric(parameters, resolution=30):

    # parameters
    a1 = parameters[0]
    a2 = parameters[1]
    a3 = parameters[2]
    e1 = parameters[3]
    e2 = parameters[4]

    # make grids
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=resolution)
    vertices_numpy = np.asarray(mesh.vertices)
    eta = np.arcsin(vertices_numpy[:, 2:3])
    omega = np.arctan2(vertices_numpy[:, 1:2], vertices_numpy[:, 0:1])

    # make new vertices
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    # reconstruct point matrix
    points = np.concatenate((x, y, z), axis=1)

    mesh.vertices = o3d.utility.Vector3dVector(points)

    return mesh

def fexp(x, p):
    return np.sign(x)*(np.abs(x + 1e-5)**p)