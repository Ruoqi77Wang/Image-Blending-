"""ectangular matrix multiplication using PyOpenCL.
"""
import sys,os
import cv2

import csv
import time

import pyopencl as cl
import pyopencl.array
import numpy as np

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue:
ctx = cl.Context(devs[0:1])
queue = cl.CommandQueue(ctx)

##list of python functions
def face_detection(_img, _grayImage, _binaryImage):
    _size=_grayImage.shape
    divisor=8
    _h, _w = _size
    minSize=(_w/divisor, _h/divisor)

    x=0
    y=0
   
    #print 'face detection: size:',_size,' h:',_h,' w:',_w,' minsize:',minSize

    #face detection
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    #rect = cv2.HaarDetectObjects(grayscale, cascade, cv.CreateMemStorage(), 1.1,  2,cv.CV_HAAR_DO_CANNY_PRUNING, (20,20))
    #result = []
    #for r in rect:
    #    result.append((r[0][0], r[0][1], r[0][0]+r[0][2], r[0][1]+r[0][3]))
    color = (255,20,0)
    faceRects = cascade.detectMultiScale(_binaryImage, 1.2, 2, cv2.CASCADE_SCALE_IMAGE,minSize)
    if len(faceRects)>0:
        for faceRect in faceRects: 
                x, y, _w, _h = faceRect
                cv2.rectangle(_binaryImage, (x, y), (x+_w, y+_h), color)
    cv2.imshow("rectangle",_binaryImage)
    print '1st coordinate: ', x,y, '  2nd coordinate:', x+_w,y, '  3rd coordinate:', x,y+_h, '  4th coordinate:', x+_w,y+_h

    mask = np.zeros(_img.shape[:2],np.uint8)
    mask[y:y+_h,x:x+_w] = 255
    res = cv2.bitwise_and(_img,_img,mask = mask)
    res = res[y:y+_h,x:x+_w]
    cv2.imshow("mask",res)

    res3 = cv2.bitwise_and(_binaryImage,_binaryImage,mask = mask)
    res3 = res3[y:y+_h,x:x+_w]
    cv2.imshow("mask22",res3)

    res4= res3.copy()
    contours, hierarchy = cv2.findContours(res4.copy(), cv2.RETR_EXTERNAL,      cv2.CHAIN_APPROX_NONE)
    size_mask = res3.shape

#we take a constraint that our face will be the contour with maximum area in image
# reference from code at: http://stackoverflow.com/questions/24731810/segmenting-license-plate-characters
    max_area = 0
    max_idx = 0

    for idx in np.arange(len(contours)):
         cnt = contours[idx]

         area = cv2.contourArea(cnt)     

         if area > max_area:
            cv2.drawContours(res4,contours,max_idx,(0,0,0),-1)
            max_area = area
            max_idx = idx
         else:
            cv2.drawContours(res4,contours,idx,(0,0,0),-1)

     
   # print 'final max_idx=', max_idx, '   final max_area=', max_area

    cv2.imshow("only face",res4)

    cv2.waitKey (0)
    cv2.destroyAllWindows() 
    return [res,res4,x,y,_h,_w]


def py_align_face(_extract,_mask,_start_x,_start_y,_width,_height):
 
    #print 'img size', _mask.shape, '   _height',_height,'  _width',_width,'  startx',_start_x,'   _starty', _start_y
    cv2.imshow("original_mask",_mask)
    #remove boundary white noise
    #_end_x = _start_x + _width -2
    #_end_y = _start_y + _height -2
    #_start_x = _start_x+1
    #_start_y = _start_y+1

    rm_x = int(0.02 * _width)
    rm_y = int(0.02 * _height)

    #print 'rmx =', rm_x,'  rmy=',rm_y

    min_starty=1
    max_endy=_height-2
    min_startx=1
    max_endx=_width-2
    
    flag=0
    for j in range(rm_y,_height-rm_y-1,1):
        for i in range(rm_x,_width-rm_x-1,1):
            #print 'mask[',j,'][',i,'] = ',_mask[j][i]
            if (_mask[j][i]==255):
             #  print 'found y at loc:',j
               min_starty = j
               flag=1
               break
        if flag == 1:
           break

    #print 'minx:',min_startx, '  miny:', min_starty,'   maxx:',max_endx,'   maxy:',max_endy
    flag = 0
    for j in range(_height-rm_y-1,rm_y,-1):
        for i in range(rm_x,_width-rm_x-1,1):
            #print 'i:',i,' j:',j
            #print 'mask[',j,'][',i,'] = ',_mask[j][i]
            if (_mask[j][i]==255):
               max_endy = j
               flag=1
               break
        if flag == 1:
           break
    flag=0    
    for i in range(rm_x,_width-rm_x-1,1):
        for j in range(rm_y,_height-rm_y-1,1):
            if (_mask[j][i]==255):
               min_startx = i
               flag=1
               break
        if flag == 1:
           break
    flag=0
    for i in range(_width-rm_x-1,rm_x,-1):
        for j in range(rm_y,_height-rm_y-1,1):
            #print 'i:',i,' j:',j
            #print 'mask[',j,'][',i,'] = ',_mask[j][i]
            if (_mask[j][i]==255):
               max_endx = i
               flag=1
               break
        if flag == 1:
           break

    #print 'minx:',min_startx, '  miny:', min_starty,'   maxx:',max_endx,'   maxy:',max_endy

    _extract = _extract[min_starty:max_endy,min_startx:max_endx]
    _mask = _mask[min_starty:max_endy,min_startx:max_endx]
    _start_x = _start_x+min_startx
    _start_y = _start_y+min_starty
    _width,_height = _mask.shape
 
    #print 'new dim',_mask.shape,'  start_x',_start_x,'   _starty:',_start_y

    cv2.imshow("aligned img",_mask)
    cv2.waitKey (0)
    cv2.destroyAllWindows() 
    return [_extract,_mask,_start_x,_start_y,_width,_height]

def scale(input_img,scale_factor,max_phase,size,components):
   coef = [[0,0,1,0,0],[-0.229,-0.0637,0.5732,0.5732,-0.0637]]
   new_size = [int(scale_factor*size[0]),size[1],components]
   new_img_py = np.zeros(new_size,input_img.dtype)
   next_k=0;
   #print 'size  = ', new_size
   #print 'inverse scale =', 1/float(scale_factor)
   #print 'step =', int(0.5+(1/float(scale_factor)))

   for j in range(0,size[1]):
       for i in  range(0,size[0],int(0.5+(1/float(scale_factor)))):
           step = int(0.5+(1/float(scale_factor)))
           appx_scale = float(int(float(scale_factor)*100))/100
           k = int(0.5+(i*appx_scale))
           if k >=new_size[0]:
              break
           next_k = int(0.5+((i+step)*appx_scale)) 
           phase = (next_k-k) if (i+step <=size[0]-1) else max_phase 
           #  l = int(0.5+(j*scale_factor))
           for p in range(0,phase):
   #            print 'j=',j,'   i=',i, '   k=',k, '   p=',p
               for c in range(0,components):

                   if i==0 :
                         new_img_py[k+p][j][c] = coef[p][2]*input_img[i][j][c] + coef[p][3]*input_img[i+1][j][c] + coef[p][4]*input_img[i+2][j][c]
                   elif i==1:
                         new_img_py[k+p][j][c] = coef[p][1]*input_img[i-1][j][c] + coef[p][2]*input_img[i][j][c] + coef[p][3]*input_img[i+1][j][c] + coef[p][4]*input_img[i+2][j][c]
                   elif i==size[0]-2:
                         new_img_py[k+p][j][c] = coef[p][0]*input_img[i-2][j][c] + coef[p][1]*input_img[i-1][j][c] + coef[p][2]*input_img[i][j][c] + coef[p][3]*input_img[i+1][j][c]
                   elif i==size[0]-1:
                         new_img_py[k+p][j][c] = coef[p][0]*input_img[i-2][j][c] + coef[p][1]*input_img[i-1][j][c] + coef[p][2]*input_img[i][j][c]
                   else:
                         new_img_py[k+p][j][c] = coef[p][0]*input_img[i-2][j][c] + coef[p][1]*input_img[i-1][j][c] + coef[p][2]*input_img[i][j][c] + coef[p][3]*input_img[i+1][j][c] + coef[p][4]*input_img[i+2][j][c]
    
   return new_img_py


def dilate(_img,components):
   size = _img.shape
   new_img = np.zeros(_img.shape,_img.dtype)
   for j in range(1,size[1]-1):
       for i in range(1,size[0]-1):
           for c in range(0,components):
                new_img[i][j][c] = max(_img[i-1][j-1][c],_img[i][j-1][c],_img[i+1][j-1][c],_img[i-1][j][c],_img[i][j][c],_img[i+1][j][c],_img[i-1][j+1][c],_img[i][j+1][c],_img[i+1][j+1][c])

   return new_img

def blend(src,mask,dst,cord,dst_mask):
    size = src.shape    
    alpha1 = 0.9
#    alpha2 = 0.4
#    print 'dst_cord', cord 
#    print 'blend_shape',src.shape
#    print 'bg_shape', dst.shape
    for j in range(0,size[0]):
       for i in range(0,size[1]):
        # print 'i=',i,'    j=',j
         for c in range(0,3):
           # print 'i=',i,'    j=',j,'    c=',c
            dst[cord[0][1] + j][cord[0][0] + i][c] = (src[j][i][c]*alpha1 +  dst[cord[0][1] + j][cord[0][0] + i][c] * (1-alpha1)) if (mask[j][i][0] == 255) else dst[cord[0][1] + j][cord[0][0] + i][c]
#            if(i>cord[0][0] and i<cord[1][0] and j>cord[0][1] and j<cord[1][1]):
#                   dst_mask[j][i][0]=127 
#
#    for j in range(0,cord[1][1]-cord[1][0] +1):
#         for i in range(0,cord[0][1]-cord[0][0] +1):
#             if(dst_mask[j][i][0] == 255):
#                  dst[cord[0][1] + j][cord[0][0] + i][c] = (src[0][0][c]*alpha2 + dst[cord[0][1] + j][cord[0][0] + i][c] * (1-alpha2)) 
    return dst 


#######################################
##list of GPU functions/parallel processes

def gpu_blend(src,mask,dst,cord,dst_mask):
     
     s = src.astype(np.uint32)
     m = mask.astype(np.uint32)
     d = dst.astype(np.uint32)
     dm = dst_mask.astype(np.uint32)
     cd = np.asarray(cord,dtype=np.int32)

     s_gpu= cl.array.to_device(queue, s)
     m_gpu = cl.array.to_device(queue,m)
     d_gpu = cl.array.to_device(queue,d)
     dm_gpu = cl.array.to_device(queue,dm)
     cd_gpu = cl.array.to_device(queue,cd)

     start = time.time()
     prg.opencl_blend(queue,s_gpu.shape,None,d_gpu.data,s_gpu.data,m_gpu.data,dm_gpu.data,cd_gpu.data,np.int32(s.shape[0]),np.int32(s.shape[1]))
     
     opencl_blending_time = time.time()-start
     print 'opencl blending time:', opencl_blending_time    
 
     cv2.imshow("opencl blended img",d_gpu.get().astype(np.uint8))

     
     cv2.waitKey (0)
     cv2.destroyAllWindows() 
     return d_gpu.get().astype(np.uint8)


def gpu_dilate(src_img,src_mask):

     ### y scaling done first

     s = src_img.astype(np.uint32)
     m=src_mask.astype(np.uint32)

     s_gpu= cl.array.to_device(queue, s)
     m_gpu= cl.array.to_device(queue, m)

     sy_gpu = cl.array.zeros(queue, src_img.shape, s.dtype)
     my_gpu = cl.array.zeros(queue, src_mask.shape, m.dtype)

     #no tiling so that we have flexibility to work on any size of image we want
     start = time.time()
     prg.opencl_dilate(queue,sy_gpu.shape,None,sy_gpu.data,s_gpu.data,np.int32(s.shape[1]),np.int32(s.shape[2]))
     prg.opencl_dilate(queue,my_gpu.shape,None,my_gpu.data,m_gpu.data,np.int32(m.shape[1]),np.int32(m.shape[2]))
     opencl_dilate_time = time.time() - start

     print 'opencl_dilate_time:',opencl_dilate_time    
 
     cv2.imshow("opencl dilated img",sy_gpu.get().astype(np.uint8))
     cv2.imshow("opencl dilated mask",my_gpu.get().astype(np.uint8))

     
     cv2.waitKey (0)
     cv2.destroyAllWindows() 
     return [sy_gpu.get().astype(np.uint8),my_gpu.get().astype(np.uint8)]

def gpu_scale(src_img,input_mask,dst_mask):

     ### y scaling done first
     _scale_factor = float(dst_mask.shape[0])/float(input_mask.shape[0])
     _max_phase = int(_scale_factor+0.5)
     _scale_factor = float(int(float(_scale_factor)*100))/100
     step = int(0.5+(1/float(_scale_factor)))

     s = src_img.astype(np.uint32)
     m=input_mask.astype(np.uint32)
     new_size_im = [int(_scale_factor*input_mask.shape[0]),input_mask.shape[1],3]
     new_size_ma = [int(_scale_factor*input_mask.shape[0]),input_mask.shape[1],1]


     s_gpu= cl.array.to_device(queue, s)
     m_gpu= cl.array.to_device(queue, m)

     cv2.imshow("opencl orig",s_gpu.get().astype(np.uint8))
     cv2.imshow("opencl orig mask",m_gpu.get().astype(np.uint8))
     sy_gpu = cl.array.zeros(queue, new_size_im, s.dtype)
     my_gpu = cl.array.zeros(queue, new_size_ma, m.dtype)

     #print 's_gpu.shape',s_gpu.shape,'  sy_gpu.shape:',sy_gpu.shape
     #print 'opencl :scale_factor:', _scale_factor,'  step:', step  
     #no tiling so that we have flexibility to work on any size of image we want
     start = time.time()
     prg.scalar(queue,sy_gpu.shape,None,sy_gpu.data,s_gpu.data,np.float32(_scale_factor),np.int32(_max_phase),np.int32(sy_gpu.shape[0]),np.int32(sy_gpu.shape[1]),np.int32(sy_gpu.shape[2]),np.int32(step),np.int32(s_gpu.shape[0]))
     prg.scalar(queue,my_gpu.shape,None,my_gpu.data,m_gpu.data,np.float32(_scale_factor),np.int32(_max_phase),np.int32(my_gpu.shape[0]),np.int32(my_gpu.shape[1]),np.int32(my_gpu.shape[2]),np.int32(step),np.int32(m_gpu.shape[0]))
     
     cv2.imshow("first scale",sy_gpu.get().astype(np.uint8))
     cv2.imshow("first scale mask",my_gpu.get().astype(np.uint8))

     s_gpu= cl.array.to_device(queue, sy_gpu.get())
     m_gpu= cl.array.to_device(queue, my_gpu.get())
     sy_t = cl.array.zeros(queue, (sy_gpu.shape[1],sy_gpu.shape[0],sy_gpu.shape[2]), m.dtype)
     my_t = cl.array.zeros(queue, (my_gpu.shape[1],my_gpu.shape[0],my_gpu.shape[2]), m.dtype)


     prg.matrix_transpose(queue,(s_gpu.shape[1],s_gpu.shape[0],s_gpu.shape[2]),None, s_gpu.data, np.int32(s_gpu.shape[0]), np.int32(s_gpu.shape[1]),np.int32(s_gpu.shape[2]), sy_t.data)
 
     prg.matrix_transpose(queue,(m_gpu.shape[1],m_gpu.shape[0],m_gpu.shape[2]),None, m_gpu.data, np.int32(m_gpu.shape[0]), np.int32(m_gpu.shape[1]),np.int32(m_gpu.shape[2]), my_t.data) 
     
     ##x_scalar starts here
     _scale_factor = float(py_dst_mask.shape[1])/float(py_src_mask.shape[1])
     _max_phase = int(_scale_factor+0.5)
     _scale_factor = float(int(float(_scale_factor)*100))/100
     step = int(0.5+(1/float(_scale_factor)))

     new_size_im = [int(_scale_factor*sy_t.shape[0]),sy_t.shape[1],3]
     new_size_ma = [int(_scale_factor*my_t.shape[0]),my_t.shape[1],1]

     s_gpu= cl.array.to_device(queue, sy_t.get())
     m_gpu= cl.array.to_device(queue, my_t.get())

     sx_gpu = cl.array.zeros(queue, new_size_im, s.dtype)
     mx_gpu = cl.array.zeros(queue, new_size_ma, m.dtype)

     #no tiling so that we have flexibility to work on any size of image we want
     prg.scalar(queue,sx_gpu.shape,None,sx_gpu.data,s_gpu.data,np.float32(_scale_factor),np.int32(_max_phase),np.int32(sx_gpu.shape[0]),np.int32(sx_gpu.shape[1]),np.int32(sx_gpu.shape[2]),np.int32(step),np.int32(s_gpu.shape[0]))
     prg.scalar(queue,mx_gpu.shape,None,mx_gpu.data,m_gpu.data,np.float32(_scale_factor),np.int32(_max_phase),np.int32(mx_gpu.shape[0]),np.int32(mx_gpu.shape[1]),np.int32(mx_gpu.shape[2]),np.int32(step),np.int32(m_gpu.shape[0]))
     
     s_gpu= cl.array.to_device(queue, sx_gpu.get())
     m_gpu= cl.array.to_device(queue, mx_gpu.get())
     sx_t = cl.array.zeros(queue, (sx_gpu.shape[1],sx_gpu.shape[0],sx_gpu.shape[2]), m.dtype)
     mx_t = cl.array.zeros(queue, (mx_gpu.shape[1],mx_gpu.shape[0],mx_gpu.shape[2]), m.dtype)

     prg.matrix_transpose(queue,(s_gpu.shape[1],s_gpu.shape[0],s_gpu.shape[2]),None, s_gpu.data, np.int32(s_gpu.shape[0]), np.int32(s_gpu.shape[1]),np.int32(s_gpu.shape[2]), sx_t.data) 
     prg.matrix_transpose(queue,(m_gpu.shape[1],m_gpu.shape[0],m_gpu.shape[2]),None, m_gpu.data, np.int32(m_gpu.shape[0]), np.int32(m_gpu.shape[1]),np.int32(m_gpu.shape[2]), mx_t.data) 
     opencl_scaling_time = time.time()-start
     print 'opencl scaling time:', opencl_scaling_time     

     cv2.imshow("opencl scaled img",sx_t.get().astype(np.uint8))
     cv2.imshow("opencl scaled mask",mx_t.get().astype(np.uint8))
     cv2.waitKey (0)
     cv2.destroyAllWindows() 
     return [sx_t.get().astype(np.uint8),mx_t.get().astype(np.uint8)]

#opencl version of code
def face_process_opencl(_img,_grayImage,_binaryImage):
    b,g,r = cv2.split(_img)
    b=b.astype(np.uint32)
    g=g.astype(np.uint32)
    r=r.astype(np.uint32)
   
    n_shape = [_img.shape[0],_img.shape[1]]
   
    cv2.imshow("img opencl b4 grayscale",_img)

    r_gpu= cl.array.to_device(queue, r)
    g_gpu= cl.array.to_device(queue, g)
    b_gpu= cl.array.to_device(queue, b)
    gray_gpu = cl.array.zeros(queue, n_shape, r.dtype)
    binary_gpu = cl.array.zeros(queue, n_shape, r.dtype)

    #launch the kernel 
    start = time.time()
    prg.grayscale(queue, n_shape , None, r_gpu.data,g_gpu.data,b_gpu.data,    gray_gpu.data,np.uint32(_img.shape[1]))
    grayscale_opencl_time=time.time()-start
    print 'opencl time in grayscale: ', grayscale_opencl_time
    gray_op = gray_gpu.get().astype(np.uint8)
    cv2.imshow("grayscale using opencl",gray_op)

    #binarization
    #opencl
    binary_gpu = cl.array.zeros(queue, n_shape, r.dtype)
    start = time.time()
    prg.binarization(queue, n_shape ,None, gray_gpu.data, binary_gpu.data, np.uint32(_img.shape[1]))
    binarization_opencl_time=time.time()-start
    print 'opencl time in binarization: ', binarization_opencl_time
    binary_op = binary_gpu.get().astype(np.uint8)
    cv2.imshow("binarization using opencl",binary_op)

#using tile
    BLOCK_DIM=16
    start = time.time()
    prg.binarizationtile(queue, n_shape ,None, gray_gpu.data, binary_gpu.data, np.uint32(w), np.uint32(h),np.uint32(BLOCK_DIM))
    opencl_tile_binarization_time=time.time()-start
    print 'opencl time (binarization using tile)', opencl_tile_binarization_time
    [a,b,c,d,e,f] = face_detection(_img, gray_op, binary_op)
    return [a,b,c,d,e,f]


# Define the OpenCL kernel you wish to run; most of the interesting stuff you
# will be doing involves modifying or writing kernels:
kernel = """
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define MAX(X,Y,Z) ((X>Y)&&(X>Z))?X:(Y>Z)?Y:Z

__kernel void grayscale(__global unsigned int *redBins, __global unsigned int *greenBins,  __global unsigned int *blueBins, __global unsigned int *grayBins, const uint size){

     unsigned int row = get_global_id(0);
     unsigned int col = get_global_id(1);
     int k;
     for(k=0;k<size+1;k++)
       grayBins[row*k+col]=0.29*redBins[row*k+col]+0.59*greenBins[row*k+col]+0.11*blueBins[row*k+col];
   
    //Gray=0.299R+0.587+0.114B
}

__kernel void binarization(__global unsigned int *grayBins, __global unsigned int *binaryBins, const uint size){

     unsigned int row = get_global_id(0);
     unsigned int col = get_global_id(1);
     int k;
     for(k=0;k<size+1;k++){
       if (grayBins[row*k+col]<=127)
         binaryBins[row*k+col]=0;
       else
         binaryBins[row*k+col]=255;
     }
    //binary: threshhold = 127
}

__kernel void binarizationtile(__global unsigned int *grayBins, __global unsigned int *binaryBins, const uint width, const uint height, unsigned BLOCK_DIM){
     //using tile
     __local float block[272];   
     unsigned int xIndex = get_global_id(0);
     unsigned int yIndex = get_global_id(1);
     unsigned int index_in = yIndex * width + xIndex;

     if((xIndex < width) && (yIndex < height))
     {
         // unsigned int index_in = yIndex * width + xIndex;
          if (grayBins[index_in]<127)
            block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = 0;
          else
            block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = 255;
      }
     barrier(CLK_LOCAL_MEM_FENCE);

     // write the transposed matrix tile to global memory
     xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
     yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);

     if((xIndex < height) && (yIndex < width))
     {
          binaryBins[index_in] = block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)];
      }


      //binary: threshhold = 127  
}

__kernel void mask(__global unsigned int *binaryBins, __global unsigned int *maskResult,const uint size,const uint widthStart, const uint widthEnd, const uint heightStart, const uint heightEnd, const uint height){

     unsigned int row = get_global_id(0);
     unsigned int col = get_global_id(1);
     float tmp = 0.0f;
     int k;
     
     for(k=0;k<height;k++)
       tmp = binaryBins[row*k+col];

     for(k=0;k<size+1;k++){

       if((row>widthStart-1 && row <widthEnd +1)&&(col>heightStart-1 && col<widthEnd+1)){
       maskResult[row*k+col] = tmp;
       }
       else
         maskResult[row*k+col] = 0;
     }
      
}

__kernel void matrix_transpose(__global unsigned int* a, const unsigned int M, const unsigned int N, const unsigned int L, __global unsigned int* y) {

    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int k = get_global_id(2);
    
    y[L*(j+M*i)+k] = a[L*(i+N*j)+k];
}

__kernel void scalar(__global unsigned int*c, __global unsigned int *a, float scale_factor, int max_phase, const unsigned int size_y, const unsigned int size_x, unsigned int components, unsigned int step, const unsigned int in_size_y) {
   
   unsigned int i = get_global_id(0);
   unsigned int j = get_global_id(1);
   unsigned int k = get_global_id(2);

   float coef[2][5] = {{0,0,1,0,0},{-0.229,-0.0637,0.5732,0.5732,-0.0637}};
   float temp = 0.0f;

   int t = ((i+0.5)/scale_factor);
    
   int min_tap = ((t>=2) * -2) + ((t<2)*-1*t);
   int max_tap = (t<=(in_size_y-3)) * 2 + (t>(in_size_y-3)) * (in_size_y-1-t);   
   int check = (i-step+0.5)/scale_factor;
   int phase = (check == t);
   
   for(int m = min_tap; m<=max_tap; m++)
       temp+=coef[phase][m+2]* a[components*(j+size_x*(t+m))+k];
   c[components*(j+size_x*i)+k] = temp; 

}
     
__kernel void opencl_dilate(__global unsigned int *c , __global unsigned int *a,unsigned int N, int comp){

   unsigned int j = get_global_id(0);
   unsigned int i = get_global_id(1);
   unsigned int k = get_global_id(2);

   unsigned int tmp1 = MAX(a[comp*(i-1 + N*(j-1)) + k],a[comp*(i + N* (j-1))+k],a[comp*(i+1 + N*(j-1))+k]);
   unsigned int tmp2 = MAX(a[comp*(i-1 + N* j)+k],a[comp*(i+ N* j)+k],a[comp*(i+1 + N*j)+k]);
   unsigned int tmp3 = MAX(a[comp*(i-1+N*(j+1))+k],a[comp*(i + N*(j+1))+k],a[comp*(i+1+N*(j+1))+k]);
 
   c[comp*(i+N*j) + k] = MAX(tmp1,tmp2,tmp3);

}

__kernel void opencl_blend(__global unsigned int *src,__global unsigned int *mask,__global unsigned int *dst,__global unsigned int *cord,__global unsigned int *dst_mask, const unsigned int M, const unsigned int N){


    unsigned int j= get_global_id(0);
    unsigned int i= get_global_id(1);
    unsigned int k = get_global_id(2);

    unsigned int tmp_src = src[3*(N*j + i)+k];
    unsigned int tmp_dst =  dst[3*((cord[1] + j)*N + cord[0] + i)+k];   
    unsigned int tmp_mask = mask[(N*j+i)];
    

    float alpha1 = 0.9;
    dst[3*((cord[1] + j)*N + cord[0] + i)+k] = tmp_src*alpha1 +  tmp_dst*(1-alpha1)*(tmp_mask == 255) + tmp_dst*(tmp_mask != 255);

}
"""

prg = cl.Program(ctx, kernel).build()

###############################################
##Stage 1: Image read
##############################################
print 'source name: miranda destination name: anna'
img = cv2.imread("faces/anna.jpg")
    #cv2.namedWindow("Image")   #create a window
cv2.imshow("destination",img)   #show image in the window

img2 = cv2.imread("faces/miranda.jpg")
cv2.imshow("source",img2)

#############################################
##Stage 2: grayscale conversion
############################################
start = time.time()
grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
python_grayscale_conversion_time = time.time()-start
print 'python_grayscale_conversion_time_destinatation img:',python_grayscale_conversion_time

start = time.time()
grayImage2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
python_grayscale_conversion_time=time.time()-start
print 'python_grayscale_conversion_time_source_img',python_grayscale_conversion_time

    #cv2.namedWindow("greyImage")
cv2.imshow("grey_python",grayImage)
size = grayImage.shape
size2 = grayImage2.shape
print 'size_dest',size,'   size_src',size2
h,w=size
h2,w2 = size2


#####################################################
##Stage 3: binarization
#####################################################
start = time.time()
thresh = 127
_,binaryImage = cv2.threshold(grayImage, thresh, 255, cv2.THRESH_BINARY)
_,binaryImage2 = cv2.threshold(grayImage2, thresh, 255, cv2.THRESH_BINARY)
binarization_time=time.time()-start
cv2.imshow("binary_python", binaryImage)
print 'python time (binarization):  ',binarization_time
    

###################################################
##Stage 4: face detection and extraction with masks
###################################################

[py_dst_extract,py_dst_mask,py_dst_startx,py_dst_starty,py_dst_width,py_dst_height] = face_detection(img, grayImage, binaryImage)
[py_src_extract,py_src_mask,py_src_startx,py_src_starty,py_src_width,py_src_height] = face_detection(img2,grayImage2,binaryImage2)
[op_dst_extract,op_dst_mask,op_dst_startx,op_dst_starty,op_dst_width,op_dst_height] = face_process_opencl(img, grayImage, binaryImage)
[op_src_extract,op_src_mask,op_src_startx,op_src_starty,op_src_width,op_src_height] = face_process_opencl(img2,grayImage2,binaryImage2)

cv2.imshow("python_face", py_src_extract)
cv2.imshow("python_mask", py_src_mask)
cv2.imshow("opencl_face", op_src_extract)
cv2.imshow("opencl_mask", op_src_mask)
########################################################################
##Stage 5: Alignement of extracts and removal of parasitic white borders
#######################################################################
[py_dst_extract,py_dst_mask,py_dst_startx,py_dst_starty,py_dst_width,py_dst_height] = py_align_face(py_dst_extract,py_dst_mask,py_dst_startx,py_dst_starty,py_dst_width,py_dst_height)
[py_src_extract,py_src_mask,py_src_startx,py_src_starty,py_src_width,py_src_height] = py_align_face(py_src_extract,py_src_mask,py_src_startx,py_src_starty,py_src_width,py_src_height)

[op_dst_extract,op_dst_mask,op_dst_startx,op_dst_starty,op_dst_width,op_dst_height] = py_align_face(op_dst_extract,op_dst_mask,op_dst_startx,op_dst_starty,op_dst_width,op_dst_height)
[op_src_extract,op_src_mask,op_src_startx,op_src_starty,op_src_width,op_src_height] = py_align_face(op_src_extract,op_src_mask,op_src_startx,op_src_starty,op_src_width,op_src_height)

#print 'py_dst_starty:',py_dst_starty,'   py_dst_startx:',py_dst_startx,'   py_dest_width:',py_dst_width,'   py_dst_height:',py_dst_height

###################################################################
##Stage 6: Source extract and mask scaling
###################################################################
### y scaling done first
scale_factor = float(py_dst_mask.shape[0])/float(py_src_mask.shape[0])
max_phase = int(scale_factor+0.5)

print 'dest_extract_size=',py_dst_mask.shape,'  src_extract_size=',py_src_mask.shape, ' scale_fact',scale_factor, ' max_phase',max_phase

py_src_mask = py_src_mask[:,:,np.newaxis]
op_src_mask = op_src_mask[:,:,np.newaxis]
py_dst_mask = py_dst_mask[:,:,np.newaxis]
op_dst_mask = op_dst_mask[:,:,np.newaxis]

start = time.time()
py_yscale_src_img = scale(py_src_extract,scale_factor,max_phase,py_src_mask.shape,3)
py_yscale_src_mask = scale(py_src_mask,scale_factor,max_phase,py_src_mask.shape,1)

#print 'yscale_op',py_yscale_img.shape
#print 'gray_img_size',gry_scale_img.shape

##x-scaling starts here
py_xscale_src_img = np.transpose(py_yscale_src_img,(1,0,2))
py_xscale_src_mask  = np.transpose(py_yscale_src_mask,(1,0,2))
#py_xscale_img = py_yscale_img
#size2 = py_xscale_img.shape
#print 'xscale_img transpose', size2

scale_factor = float(py_dst_mask.shape[1])/float(py_src_mask.shape[1])
max_phase = int(scale_factor+0.5)
py_post_scale_img = scale(py_xscale_src_img,scale_factor,max_phase,py_xscale_src_mask.shape,3)
py_post_scale_mask = scale(py_xscale_src_mask,scale_factor,max_phase,py_xscale_src_mask.shape,1) 

#print 'post_scale_img',py_xscale_img_post.shape
py_final_src_img = np.transpose(py_post_scale_img,(1,0,2))
py_final_src_mask = np.transpose(py_post_scale_mask,(1,0,2))
#print 'size_img',final_img.shape
python_scaling_time = time.time()-start
print 'python image scaling time:',python_scaling_time

cv2.imshow("scaled img",py_final_src_img)
cv2.imshow("scaled mask",py_final_src_mask)

[op_final_src_img,op_final_src_mask] = gpu_scale(op_src_extract,op_src_mask,op_dst_mask)

#############################################################
##Stage 7: dilate image to remove scaling discontinuous artifacts
###############################################################
start = time.time()
py_dilated_img = dilate(py_final_src_img, 3)
py_dilated_mask = dilate(py_final_src_mask,1)
python_dilation_time = time.time()-start
print 'python dilation time:',python_dilation_time


cv2.imshow("py_dilated_img:",py_dilated_img)
cv2.imshow("py_dilated_mask:",py_dilated_mask)

[op_dilated_src,op_dilated_mask] = gpu_dilate(op_final_src_img,op_final_src_mask)


##################################################################
##Stage 8: Final alpha blending
################################################################
img_op = img
py_dmask_cord = [[py_dst_startx,py_dst_starty],[py_dst_startx+py_dst_width-1,py_dst_starty+py_dst_height-1]]
op_dmask_cord = [[op_dst_startx,op_dst_starty],[op_dst_startx+op_dst_width-1,op_dst_starty+op_dst_height-1]]

start = time.time()
py_final_blend = blend(py_dilated_img,py_dilated_mask,img,py_dmask_cord,py_dst_mask)
final_blend_time = time.time()-start
print 'python final blending time:',final_blend_time


op_final_blend = gpu_blend(op_dilated_src,op_dilated_mask,img_op,op_dmask_cord,op_dst_mask)
cv2.imshow("python final blend",py_final_blend)
cv2.imshow("opencl final blend",op_final_blend)

cv2.waitKey (0)
cv2.destroyAllWindows() 


