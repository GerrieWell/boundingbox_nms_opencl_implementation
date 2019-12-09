import nms_opencl 
import numpy as np
bbox=np.load("../bbox.npy")
n_bbox = bbox.shape[0]
keep_out=np.zeros(bbox.shape[0],dtype=np.int32)
n=nms_opencl.nms_opencl(keep_out,bbox)
np.savetxt("keep_out.csv", keep_out,fmt='%d')
print(n,keep_out)