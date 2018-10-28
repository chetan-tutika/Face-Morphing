'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: source image
    - Input im2: target image
    - Input im1_pts: correspondences coordiantes in the source image
    - Input im2_pts: correspondences coordiantes in the target image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imageio
from PIL import Image
import numpy as np
from scipy.spatial import Delaunay
from numpy.linalg import inv
import cv2

def getABC(im, im_pts, tri):
  xx_im, yy_im = np.meshgrid(np.arange(0, im.shape[0]),np.arange(0, im.shape[1]))
  #xx_im2, yy_im2 = np.meshgrid(np.arange(0, im2.shape[1]),np.arange(0, im2.shape[0]))
  #print(xx_im.shape, im.shape)

  #im_index = np.stack((xx_im, yy_im),axis = -1)
  im_index = np.stack((xx_im, yy_im),axis = -1)

  #im2_index = np.stack((xx_im2, yy_im2),axis = -1)
  #print('im1_index',im_index.reshape((-1,2)[:3]))

  im_each_tri  = tri.find_simplex(im_index.reshape((-1,2)))
  tri_vertex = tri.simplices
  tri_vertex_each_point_im = tri_vertex[im_each_tri]
  tri_vertex_eachStrong_point_im = im_pts[tri_vertex]


  #print('tri_vertex_each_point_im1', tri_vertex_each_point_im1[:2])

  tri_vertex_each_point_im_ABC = im_pts[tri_vertex_each_point_im]
  #tri_vertex_each_point_im2_ABC = im2_pts[tri_vertex_each_point_im2]
  #print('tri_vertex_each_point_im1', tri_vertex_each_point_im1_ABC[:2])
  tri_vertex_each_Strongpoint_im_ABC_x = tri_vertex_eachStrong_point_im[:,:,0].reshape((-1,1,3))
  tri_vertex_each_Strongpoint_im_ABC_y = tri_vertex_eachStrong_point_im[:,:,1].reshape((-1,1,3))

  #tri_vertex_each_point_im_ABC_shape = tri_vertex_each_point_im1_ABC.shape
  #tri_vertex_each_point_im_ABC_x = tri_vertex_each_point_im_ABC[:,:,0].reshape((-1,1,3))
  #tri_vertex_each_point_im_ABC_y = tri_vertex_each_point_im_ABC[:,:,1].reshape((-1,1,3))
  #print('tri_vertex_each_point_im_ABC_y',tri_vertex_each_point_im_ABC_y.shape[0])
  a = tri_vertex_each_Strongpoint_im_ABC_y.shape

  ones_shape = np.ones(a)
  #print(ones_shape.shape)
  tri_vertex_each_point_im_ABC1 = np.hstack((tri_vertex_each_Strongpoint_im_ABC_x, tri_vertex_each_Strongpoint_im_ABC_y, ones_shape))
  tri_vertex_each_point_im_ABC1_inv = inv(tri_vertex_each_point_im_ABC1)
  #b = im_index.reshape(-1,2)
  #ones_shape1 = b.shape[0]
  tri_vertex_a_each_point_im_ABC1_inv = tri_vertex_each_point_im_ABC1_inv[im_each_tri]

  im_index_1 = np.hstack((im_index.reshape((-1,2)), np.ones((im_index.shape[0]*im_index.shape[1],1))))
  im_index_1 = im_index_1.reshape((-1,1,3))
  #im_index_1 = np.swapaxes(im_index_1,1,2)
  im_index_1 = np.tile(im_index_1,[1,3,1])
  #print(im_index_1[:2])

  #ABC1 = np.sum((tri_vertex_each_point_im_ABC1_inv * im_index_1).reshape((-1,3)),axis = 1)
  ABC1 = np.sum((tri_vertex_a_each_point_im_ABC1_inv * im_index_1),axis = 2)
  #print(ABC1.shape)

  #ABC1 = ABC1.reshape(-1,3,1)
  #ABC1 = np.sum((tri_vertex_each_point_im_ABC1_inv * im_index_1),axis = 2)

  # tri_vertex_each_point_im1_ABC1 = np.hstack((tri_vertex_each_point_im1_ABC.reshape((-1, 2)), np.ones((ones_shape[0], 1))))
  # tri_vertex_each_point_im1_ABC1 = tri_vertex_each_point_im1_ABC1.reshape((90000,3,3))

  # tri_vertex_each_point_im2_ABC1 = np.hstack((tri_vertex_each_point_im2_ABC.reshape((-1, 2)), np.ones((ones_shape[0], 1))))
  # tri_vertex_each_point_im2_ABC1 = tri_vertex_each_point_im2_ABC1.reshape((90000,3,3))



  #print(len(tri_vertex_each_point_im1_ABC1>-1))
  # print(tri_vertex_each_point_im_ABC1_inv.shape)
  # print(im_index_1.shape)
  # print(ABC1[:3])



  return ABC1, tri_vertex_each_point_im

def AverageShape(im, im_pts, each_tri_vertex,tri, ABC):
  tri_index_each_point = im_pts[each_tri_vertex]
  #print(tri_index_each_point[:3])
  tri_index_each_point_ABC_x = tri_index_each_point[:,:,0].reshape((-1,1,3))
  tri_index_each_point_ABC_y = tri_index_each_point[:,:,1].reshape((-1,1,3))
  a = tri_index_each_point_ABC_y.shape
  ones_shape = np.ones(a)
  #print(ones_shape.shape)
  tri_index_each_point_ABC1 = np.hstack((tri_index_each_point_ABC_x, tri_index_each_point_ABC_y, ones_shape))
  ABC = ABC.reshape((-1,1,3))
  #ABC = np.swapaxes(ABC, 1, 2)
  ABC_tile = np.tile(ABC,[1,3,1])
  #XY1 = np.sum((tri_index_each_point_ABC1 * ABC_tile).reshape((-1,3)),axis = 1)
  XY1 = np.sum((tri_index_each_point_ABC1 * ABC_tile),axis = 2)

  #XY1 = XY1.reshape(-1,3,1)
  #XY1 = XY1.reshape(-1,1,3)
  XY1_x = np.clip(np.around(XY1[:,0]), 0, im.shape[1]-1).astype(int)
  XY1_y = np.clip(np.around(XY1[:,1]), 0, im.shape[0]-1).astype(int)
  #XY1_x = XY1_x.reshape(im.shape[:2]).astype(int)
  #XY1_y = XY1_y.reshape(im.shape[:2]).astype(int)

  xx_im, yy_im = np.meshgrid(np.arange(0, im.shape[0]),np.arange(0, im.shape[1]))
  im1 = im.copy()
  im2 = im.copy()

  im1[[yy_im.flatten()],[xx_im.flatten()]] = im2[[XY1_y], [XY1_x]]
  #im1[[xx_im.flatten()],[yy_im.flatten()]] = im[[XY1_x.flatten()], [XY1_y.flatten()]]

  #plt.imshow(im1)
  #plt.show()

  #print(XY1.shape)
  #print(XY1_x)
  #print(XY1_y)

  return im1




def Delaunay_tri(im1_pts, im2_pts, warp_frac, dissolve_frac):
  Average_index = (warp_frac*im2_pts + (1 - warp_frac)*im1_pts).astype(int)
  tri = Delaunay(Average_index)
  # plt.figure(num = 'Average')
  # plt.triplot(Average_index[:,0], Average_index[:,1], tri.simplices)
  # plt.plot(Average_index[:,0], Average_index[:,1], 'o')  
  # plt.show()
  tri_vertex = tri.simplices
  return tri, Average_index


def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  # TODO: Your code here
  # Tips: use Delaunay() function to get Delaunay triangulation;
  # Tips: use tri.find_simplex(pts) to find the triangulation index that pts locates in.
  imC1 = im1.copy()
  imC2 = im2.copy()
  morphed_im = np.zeros((len(warp_frac), im1.shape[0],im1.shape[1],im1.shape[2]))
  #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (imC1.shape[1],imC1.shape[0]), True)
 
  #im1_pts = np.flip(im1_pts,1)
  #im2_pts = np.flip(im2_pts,1)
  list_images = []
  k =0
  for dissolve_frac_i, warp_frac_i in zip(dissolve_frac, warp_frac):
    #print(warp_frac_i)

    tri, Average_index = Delaunay_tri(im1_pts, im2_pts, warp_frac_i, dissolve_frac)

    ABC, each_tri_vertex = getABC(imC1, Average_index, tri)
    #print('ABC',ABC.shape)

    im3 = AverageShape(imC1, im1_pts, each_tri_vertex, tri, ABC)
    im4 = AverageShape(imC2, im2_pts, each_tri_vertex, tri, ABC)

    morphed_im[k,:,:,:] = (dissolve_frac_i*im4 + (1 - dissolve_frac_i)*im3)#.astype('uint8')
    #vidw = (dissolve_frac_i*im4 + (1 - dissolve_frac_i)*im3)
    k = k+1
    #plt.imshow(im_Average)
    #result = Image.fromarray((morphed_im).astype(np.uint8))
    #result.save('blended_Image{}.bmp'.format(warp_frac_i))
    #list_images.append([k,morphed_im])
    #plt.show()
    #out.write(vidw)

  #print(tri_vertex)
  #out.release()

  return morphed_im

if __name__ == "__main__":

  import time
  start_time = time.time()
  img1 = np.array(Image.open('Face.jpg').convert('RGB'))
  img2 = np.array(Image.open('Damon.jpg').convert('RGB'))
  wf = np.linspace(0,1,num = 60, dtype =float)
  df = np.linspace(0,1,num = 60, dtype = float)


  im1_pts = np.load('im1_cor.npy')
  im2_pts = np.load('im2_cor.npy')
  #mmm
  #im1_pts = np.flip(im1_pts,1)
  #im2_pts = np.flip(im2_pts,1)

  #im1_pts = np.clip(np.around(im1_pts),0, img1.shape[:2]).astype(int)
  #im2_pts = np.clip(np.around(im2_pts),0, img2.shape[:2]).astype(int)

  #print(len(im1_pts),len(im2_pts))
  res_list = []

  #morphed_set = morph_tri(img1, img2, im1_pts, im2_pts, np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]),[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
  morphed_set = morph_tri(img1, img2, im1_pts, im2_pts, wf,df)

  k = 0
  while k < morphed_set.shape[0]:
    res_list.append(morphed_set[k, :, :, :])
    k += 1

  # generate gif file
imageio.mimsave('./eval_out.gif', res_list)



