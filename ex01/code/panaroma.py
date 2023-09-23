
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from IPython.display import HTML
from matplotlib import animation
from matplotlib.patches import Polygon
import cv2
# Load images
img1 = cv2.cvtColor(cv2.imread('./image-1.jpg'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('./image-2.jpg'), cv2.COLOR_BGR2RGB)
print(img1.shape)
# Load matching points
npz_file = np.load('./panorama_points.npz')
points_source = npz_file['points_source']
points_target = npz_file['points_target']

'''
# Let's visualize the images
f = plt.figure(figsize=(15, 5))
ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax1.imshow(img1)
ax2.imshow(img2)
'''
def draw_matches(img1, points_source, img2, points_target):
    ''' Returns an image with matches drawn onto the images.
    '''
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    output_img = np.zeros((max([r, r1]), c + c1, 3), dtype='uint8')
    output_img[:r, :c, :] = np.dstack([img1])
    output_img[:r1, c:c + c1, :] = np.dstack([img2])

    for p1, p2 in zip(points_source, points_target):
        (x1, y1) = p1[:2]
        (x2, y2) = p2[:2]

        cv2.circle(output_img, (int(x1), int(y1)), 10, (0, 255, 255), 10)
        cv2.circle(output_img, (int(x2) + c, int(y2)), 10, (0, 255, 255), 10)

        cv2.line(output_img, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 255, 255), 5)

    return output_img

'''
f = plt.figure(figsize=(20, 10))
vis = draw_matches(img1, points_source[:5], img2, points_target[:5])
plt.imshow(vis)
'''

#Using DLT
def get_Ai(xi_vector, xi_prime_vector):
    assert(xi_prime_vector.shape==(3,) and xi_vector.shape==(3,))
    zero_vector = np.zeros((3,),dtype=np.float32)
    xi,yi,wi=xi_prime_vector
    Ai=np.stack([
        np.concatenate([zero_vector,-wi*xi_vector,yi*xi_vector]),
        np.concatenate([wi*xi_vector,zero_vector,-xi*xi_vector])
    ])
    assert(Ai.shape==(2,9))
    return Ai

def get_A(points_source,points_target):
    assert(points_source.shape[-1]==3 and points_target.shape[-1]==3)
    N=points_source.shape[0]
    corres=zip(points_source,points_target)
    A=np.concatenate([get_Ai(p1,p2)for (p1,p2)in corres])
    assert(A.shape == (2*N,9))
    return A


#Direct Linear Transform
def get_homography(points_source, points_target):
    A = get_A(points_source,points_target)
    u, s, vh = np.linalg.svd(A)
    H = vh[-1].reshape(3, 3)
    H = H/H[2,2]
    return H
#stiching~
def stich_images(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min,-y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img
    
H = get_homography(points_target, points_source)
stiched_image = stich_images(img1, img2, H)
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Stiched Panorama")
plt.imshow(stiched_image)
plt.show()


def get_keypoints(img1, img2):
    orb = cv2.ORB_create(nfeatures=2000)

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    p_source = np.float32([ keypoints1[good_match.queryIdx].pt for good_match in good ]).reshape(-1,2)
    p_target = np.float32([ keypoints2[good_match.trainIdx].pt for good_match in good ]).reshape(-1,2)
    N = p_source.shape[0]
    p_source = np.concatenate([p_source, np.ones((N, 1))], axis=-1)
    p_target = np.concatenate([p_target, np.ones((N, 1))], axis=-1)
    return p_source, p_target

