import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from IPython.display import HTML
from matplotlib import animation
from matplotlib.patches import Polygon
import cv2
H, W = 128, 128
def get_cube(center = (0,0,2),rotation_angles=[0.,0.,0.],with_normals=False,scale=1.):
    corners = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
    corners = corners - np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3)
    corners = corners * scale
    rot_mat = R.from_euler('xyz', rotation_angles, degrees=True).as_matrix()
    corners = np.matmul(corners, rot_mat.T)
    #corners = np.matmul(corners,rot_mat.T)
    corners = corners + np.array(center, dtype=np.float32).reshape(1, 3)
    faces = np.array([
    # all faces containing (0, 0, 0)
    [corners[0], corners[1], corners[3], corners[2]],
    [corners[0], corners[1], corners[5], corners[4]],
    [corners[0], corners[2], corners[6], corners[4]],
    # all faces containing (1, 1, 1)
    [corners[-1], corners[-2], corners[-4], corners[-3]],
    [corners[-1], corners[-2], corners[-6], corners[-5]],
    [corners[-1], corners[-3], corners[-7], corners[-5]],
    ])

    if with_normals:
        normals = np.array([(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        normals = np.matmul(normals, rot_mat.T)
        return faces, normals
    else:
        return faces

def get_camera_intrinsics(fx=70, fy=70, cx=W/2., cy=H/2.):
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0,  1],], dtype=np.float32)
    assert(K.shape == (3, 3) and K.dtype == np.float32)
    return K    

def get_perspective_projection(x_c, K):
    assert(x_c.shape == (3,) and K.shape == (3, 3))
    projected = np.matmul(K,x_c)
    x_s = projected[:2]/projected[-1]
    assert(x_s.shape == (2,))
    return x_s

def project_cube(cube, K):
    s = cube.shape
    assert(s[-1] == 3)
    cube = cube.reshape(-1, 3)
    projected_cube = np.stack([get_perspective_projection(p, K) for p in cube])
    projected_cube = projected_cube.reshape(*s[:-1], 2)#除了最后一个全部取
    return projected_cube

def plot_projected_cube(projected_cube, figsize=(5, 5), figtitle=None, colors=None, face_mask=None):
    assert(projected_cube.shape == (6, 4, 2))
    fig, ax = plt.subplots(figsize=figsize)
    if figtitle is not None:
        fig.suptitle(figtitle)
    if colors is None:
        colors = ['C0' for i in range(len(projected_cube))]
    if face_mask is None:
        face_mask = [True for i in range(len(projected_cube))]

    ax.set_xlim(0, W), ax.set_ylim(0, H)
    ax.set_xlabel('Width'), ax.set_ylabel("Height")

    for (cube_face, c, mask) in zip(projected_cube, colors, face_mask):
        if mask:
            ax.add_patch(Polygon(cube_face, color=c))
    plt.show()



def get_face_color(normal, point_light_direction=(0, 0, 1)):
    assert(normal.shape == (3,))
    point_light_direction = np.array(point_light_direction, dtype=np.float32)
    light_intensity = np.sum(normal * (-point_light_direction))
    color_intensity = 0.1 + (light_intensity * 0.5 + 0.5) * 0.8
    color = np.stack([color_intensity for i in range(3)])
    return color

def get_face_colors(normals, light_direction=(0, 0, 1)):
    colors = np.stack([get_face_color(normal, light_direction) for normal in normals])
    return colors
def get_face_mask(cube, normals, camera_location=(0, 0, 0)):
    assert(cube.shape == (6, 4, 3) and normals.shape[-1] == 3)
    camera_location = np.array(camera_location).reshape(1, 3) 

    face_center = np.mean(cube, axis=1)
    viewing_direction = camera_location - face_center
    dot_product = np.sum(normals * viewing_direction, axis=-1)
    mask = dot_product > 0.0
    return mask

cube, normals = get_cube(rotation_angles=[30, 50, 0], with_normals=True)
colors = get_face_colors(normals)
mask = get_face_mask(cube, normals)
projected_cube = project_cube(cube, get_camera_intrinsics())
plot_projected_cube(projected_cube, figtitle="Projected Cuboid with Shading", colors=colors, face_mask=mask)

