import _init_paths
import caffe
#from scipy.ndimage.morphology import grey_dilation, grey_erosion

import cv2
import pyrealsense as pyrs
import numpy as np
from scipy.stats import threshold

lut = np.uint8(np.zeros((3,256)))
lut[:,0] = [50,50,50] #background
lut[:,7] = [255,0,0] #car
lut[:,15] = [0,0,255] #person
lut[:,16] = [0,255,0] #potted plant

min_cnt_moment = 50 #min mass of contour to be considered

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        pyrs.start()
        caffe.set_mode_gpu()
        self.dev = pyrs.Device()
        self.dev.set_device_option(pyrs.constants.rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 1)
        self.dev.set_device_option(pyrs.constants.rs_option.RS_OPTION_R200_LR_AUTO_EXPOSURE_ENABLED, 1)
        print 'start net init'
        self.net = caffe.Net('/home/rafi/test_fcn/fcn.berkeleyvision.org-master/voc-fcn32s/deploy.prototxt', 
                    '/home/rafi/test_fcn/fcn.berkeleyvision.org-master/voc-fcn32s/fcn32s-heavy-pascal.caffemodel', caffe.TEST)
                    
        print 'finished net init'

        # shape for input (data blob is N x C x H x W), set data
    
    def __del__(self):
        pyrs.stop()
        #self.video.release()
        pass
    
    def get_frame(self):
        self.dev.wait_for_frame()
        #print 'got frame'
        #grab color and depth-aligned-to-color
        color = self.dev.colour
        image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        depth = self.dev.dac * self.dev.depth_scale * 1000 #this makes d be the depth in mm
#        depth = grey_dilation(depth, size=(9,9)) #remove noise of unmeasured pixels that show up as 0's.
        #prepare color for caffe
        in_ = np.array(image, dtype=np.float32)
        #in_ = in_[:,:,::-1] #using cv2.imread, we are already BGR
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))
        self.net.blobs['data'].reshape(1, *in_.shape)
        self.net.blobs['data'].data[...] = in_
        #print 'start net forward'
        self.net.forward()
        #print 'finished net forward'
        out = self.net.blobs['score'].data[0].argmax(axis=0)
        #keep only 'person' class
        pre_thresh = np.empty_like(out, dtype=np.uint8)
        np.copyto(pre_thresh, out, casting='unsafe')
        thresh = threshold(pre_thresh, threshmin = 15, threshmax = 15, newval = 0) #throw out all values that are too small or too large
        thresh = threshold(thresh, threshmax = 1, newval = 255) #make remaining values 255
        thresh = thresh.astype(np.uint8) 
        #erode that a lot to make sure we focus on center
        kernel = np.ones((32,32), np.uint8)
        eroded_thresh = cv2.erode(thresh, kernel, iterations = 1)
        #draw its contour on image
        _, contours, _ = cv2.findContours(eroded_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        best_cnt_center = None
        best_cnt_depth = None
        best_cnt = None
        for cnt in contours:
            M = cv2.moments(cnt)
            if (M['m00'] < min_cnt_moment):
                continue
            center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            mask = np.zeros(depth.shape, np.uint8)   
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            cnt_depth = cv2.bitwise_and(depth, depth, mask=mask)            
            depth_hist, bin_edges = np.histogram(cnt_depth, 930, (700, 10000)) #calculate the histogram of values between 70cm and 10m
            likely_depth = np.argmax(depth_hist) * 10 + 700 #the bin with the most values
            #avg_depth = depth[center[1], center[0]]
            if best_cnt_depth is None or likely_depth < best_cnt_depth:
                best_cnt_depth = likely_depth
                best_cnt_center = center
                best_cnt = cnt
        if best_cnt_depth is not None:
            #print "best_cnt_depth ", best_cnt_depth
            cv2.drawContours(image, [best_cnt], 0, (0,255,0), 3)
            cv2.circle(image, best_cnt_center, 5, [0,0,255], 2)
            cv2.putText(image, 'Depth: {:2.0f}'.format(best_cnt_depth), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
#        out_8 = np.empty_like(out, dtype=np.uint8)
#        np.copyto(out_8, out, casting='unsafe')
#        out_clr = np.zeros((out_8.shape[0],out_8.shape[1],3))
#        out_clr[:,:,0] = cv2.LUT(out_8,lut[0])
#        out_clr[:,:,1] = cv2.LUT(out_8,lut[1])
#        out_clr[:,:,2] = cv2.LUT(out_8,lut[2])

        #print image.shape, dimg.shape
        d_uint = np.array(depth / 10000 * 256, np.uint8) #look at depth up to 10 meters
        dimg = cv2.applyColorMap(cv2.cvtColor(d_uint, cv2.COLOR_GRAY2BGR), cv2.COLORMAP_RAINBOW)
        cd = np.concatenate((image,dimg), axis=0)
        #success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', cd)
        return jpeg.tobytes()