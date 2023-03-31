#!/usr/bin/env python3

from cmath import atan, sin
from email.mime import image
from re import X
from turtle import width
from xml.etree.ElementTree import PI
import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
import imutils
import time
import cv2
import numpy as np
import warnings
import math

#suppress warnings
warnings.filterwarnings('ignore')
from os.path import expanduser
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2

from rostopic import get_topic_type
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo  
from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes, FourPoints
from std_msgs.msg import String, Float64
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs2
import queue

# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import random

def solvePNP(image_points_3D,image_points_2D,im0):
    size = im0.shape
    focal_length = size[1]
    distortion_coeffs = np.zerofind_plane(([0][0]), int(image_points_2D[0][1]))
    point2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))       
    cv2.line(im0, point1, point2, (255,255,255), 1) 
    vector_rotation = vector_rotation*180/3.14
    #im0 = cv2.putText(im0, "RY:{}".format(int(vector_rotation[1])), point1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
    print("success: {}, \nVec translation: {}\nVEC ROTATIONNN: {}\n {}".format(success,vector_translation,vector_rotation,int(vector_rotation[1])))
    
    return vector_rotation, vector_translation

def random_points_generator(num_points, min_x, max_x, min_y, max_y):  #Generates a list of random 2D points within the specified range.
    points = []
    for i in range(num_points):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        points.append((x, y))
    return points
    
@torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        print("Initialise detector")
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        self.view_image = rospy.get_param("~view_image")
        # FPS update time in seconds
        self.zplay_time = 2
        self.fc = 0
        self.FPS = 0
        # Initialize weights 
        weights = rospy.get_param("~weights")
        self.button = rospy.get_param("~button")
        self.start_time = time.time()

         # Initialize model
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
        self.unpack() #Unpack self.model attributes
        self.avglen = 30
        #self.store = {np.zeros((len(self.names),0)) }
        self.store={} 
        print("Initialised store:",self.store)
        #print(self.store)
         #Setting inference size
        self.img_size = [rospy.get_param("~inference_size_h", 640), rospy.get_param("~inference_size_w",480)]
        print(self.img_size)
        self.img_size = check_img_size(self.img_size, s=self.stride)
        print(self.img_size)

        self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.img_size))

        #self.store = []   
        #Initlaise CV bridge
        self.bridge = CvBridge()

        # Initialize subscribers
        input_image_topic = rospy.get_param("~input_image_topic") #color image
        print("Input topic is ", input_image_topic)
        rospy.Subscriber(input_image_topic, msg_Image, self.callback)
        self.intrinsics =  None #Variable to store camera params
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.cameraInfoCallback) #camera_info
        rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.depthInfoCalback) #camera_info
        rospy.Subscriber('/camera/depth/image_rect_raw', msg_Image, self.imageDepthCallback)
        
        
        #Initialise Publishers
        self.pred_pub = rospy.Publisher(rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10 )
        self.publish_image = rospy.get_param("~publish_image") #Output image
        
        print("Initialisation complete")
    
    def preprocess(self, img):
            """
            Adapted from yolov5/utils/datasets.py LoadStreams class
            """
            img0 = img.copy()
            img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
            # Convert
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)

            return img, img0 

    def callback(self, data):
        #Get image
        im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8") #(720, 1280, 3) img
        print("Color image shape:",im.shape)
        im, im0 = self.preprocess(im) #shape = (1, 3, 384, 640)
        # Run inference
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            print("In here")
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )
        # Process predictions 
        det = pred[0].cpu().numpy()
        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header 
        bounding_boxes.image_header = data.header
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from im size to im0 size (1, 3, 384, 640) -> (720, 1280, 3) 
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                self.populate(bounding_box,int(cls),conf,xyxy)
                points = Pixel_points(bounding_box)
                points.converge(0)
                #image_points_2D = np.array([ points.center,points.leftlow, points.righthigh,  points.rightlow, points.lefthigh, (points.center[0],points.center[1]+30)], dtype="double")
                image_points_2D = np.array(random_points_generator(30,bounding_box.xmin,bounding_box.xmax,bounding_box.ymin,bounding_box.ymax))

                
                #(bounding_box.x,bounding_box.y) = points.center
                depth_ratio_x = 480/720 #(720,1280) - aligned, (480, 848) depth_raw
                depth_ratio_y = 848/1280 #(720,1280) - aligned, (480, 848) depth_raw

                depth_center = self.depth_image[int(points.center[1]*depth_ratio_y),int(points.center[0]*depth_ratio_x)] 
                bounding_box.z= depth_center  
                bounding_box.x,bounding_box.y,bounding_box.z = rs2.rs2_deproject_pixel_to_point(self.depth_intrinsics, [points.center[0], points.center[1]], depth_center)
                im0 = cv2.putText(im0, "({},{},{})".format(int(bounding_box.x),int(bounding_box.y),int(bounding_box.z)), (int(points.center[0]),int(points.center[1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2, cv2.LINE_AA)
                #3D Points 
                image_points_3D = []
                image_points_2D_filtered = []
                image_points_3D
                for point in image_points_2D:
                    depth = self.depth_image[int(point[1]*depth_ratio_y),int(point[0]*depth_ratio_x)]             
                    if depth > 20:
                        #point_3d = (point[0],point[1],depth)
                        point_3d = rs2.rs2_deproject_pixel_to_point(self.depth_intrinsics, [point[0], point[1]], depth)
                        image_points_2D_filtered.append(point)
                        image_points_3D.append(point_3d)    
                        
                print("Number of points for {}:{}".format(self.names[int(cls)], len(image_points_3D)))
                image_points_3D = np.array(image_points_3D)
                image_points_2D_filtered = np.array(image_points_2D_filtered, dtype=np.float32)

                #Displaying the 2D points
                for p in image_points_2D_filtered:
                    cv2.circle(im0, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                                          
                #Solve PnP
                #vector_rotation, vector_translation = solvePNP(image_points_3D,image_points_2D,im0)
                  # Estimate the pose of the object in the image
                image_points_2D_filtered = np.array(image_points_2D_filtered, dtype=np.float32)
                #try:
                success, rvec, tvec = cv2.solvePnP(image_points_3D, image_points_2D_filtered, self.K, distCoeffs = self.distCoeff)
                # Convert the rotation vector to a rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                # Extract the rotational Y component of the rotation matrix
                rotational_y =  np.rad2deg(np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2]))

                print('Rotational Y:', rotational_y)

                rvec_deg = np.rad2deg(rvec)
                print(self.names[int(cls)],rvec_deg, tvec)
                #im0 = cv2.putText(im0, "RY:{}".format(int(rotational_y)), (int(points.center[0]),int(points.center[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
            
                # except:
                #     print("Not enough points with depth points for object ",self.names[int(cls)])
               # print(bounding_box)
                  #Angle detection using SVD 
                if len(image_points_3D) > 2:
                    param = self.find_plane(image_points_3D)
                    alpha = math.atan(param[2]/param[0])*180/math.pi
                    if(alpha < 0):
                        alpha = alpha + 90
                    else:
                        alpha = alpha - 90

                    gamma = math.atan(param[2]/param[1])*180/math.pi
                    if(gamma < 0):
                        gamma = gamma + 90
                    else:
                        gamma = gamma - 90
                
                
                    current_store = self.store.get(str(cls),[])
                    if len(current_store) >= self.avglen:
                        current_store = np.delete(current_store,0)    
                    new_store = np.append(current_store,rotational_y)
                    self.store[str(cls)] = new_store
                    #print("Final:",self.store[str(cls)])
                    #print("no of items:",len(self.store[str(cls)]))
                    average = sum(new_store)/len(new_store)
                    #print("Ry: {} Rx: {}, center {}".format(alpha,gamma,points.center))
                    bounding_box.ry = average
                    im0 = cv2.putText(im0, "RY:{}".format(int(average)), (int(points.center[0]),int(points.center[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
                else:
                    im0 = cv2.putText(im0, "Not enough depth information", (int(points.center[0]),int(points.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
                bounding_boxes.bounding_boxes.append(bounding_box)
                # Annotate the image
                label = f"{self.names[int(cls)]} {depth_center:.2f}"# {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(int(cls), True)) 

        # Publish prediction
        self.pred_pub.publish(bounding_boxes)
        #Calculate fps
        self.fc+=1
        TIME = time.time() - self.start_time

        if (TIME) >= self.zplay_time :
            self.FPS = self.fc / (TIME)
            self.fc = 0
            self.start_time = time.time()

        fps_zp = "FPS: "+str(self.FPS)[:5]
        
        im0 = cv2.putText(im0, str(fps_zp), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        cv2.imshow(str(0), im0) 
        cv2.imshow("depth",self.depth_image)     
        cv2.waitKey(1) 
    
    def find_plane(self,points):

        c = np.mean(points, axis=0)
        r0 = points - c
        u, s, v = np.linalg.svd(r0)
        nv = v[-1, :]
        ds = np.dot(points, nv)
        param = np.r_[nv, -np.mean(ds)]
        return param

    def populate(self,bounding_box,c,conf,xyxy):
        bounding_box.Class = self.names[c]
        bounding_box.probability = conf 
        bounding_box.xmin = int(xyxy[0])
        bounding_box.ymin = int(xyxy[1])
        bounding_box.xmax = int(xyxy[2])
        bounding_box.ymax = int(xyxy[3])

    def unpack(self):
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine,)
    
    def imageDepthCallback(self, data):
        self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding) #(720,1280) - aligned, (480, 848) depth_raw
        print("Depth image shape:",self.depth_image.shape)
    
    def depthInfoCalback(self,data):
        self.depth_intrinsics = rs2.intrinsics()
        self.depth_intrinsics.width = data.width
        self.depth_intrinsics.height = data.height
        self.depth_intrinsics.fx = data.K[0]
        self.depth_intrinsics.fy = data.K[4]
        self.depth_intrinsics.ppx = data.K[2]
        self.depth_intrinsics.ppy = data.K[5]
        self.depthK = np.array(data.K).reshape(3, 3)
        print("Depth info:",data.width,data.height)

    def cameraInfoCallback(self, cameraInfo):
        self.K = np.array(cameraInfo.K).reshape(3, 3)
        self.distCoeff = cameraInfo.D
        print("Camera K is: ",self.K)
        print("Camera distCoeff is: ",self.distCoeff)
        print(cameraInfo.width,cameraInfo.height)
        #print("Camra info")
        try:
            # import pdb; pdb.set_trace()
            if self.intrinsics:
                return
            print("Storing camera info")
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equiztant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
            print(self.intrinsics.width,self.intrinsics.height)
        except CvBridgeError as e:
            print(e)
            return

class Pixel_points:
    def __init__(self,bounding_box):
        self.center = (bounding_box.xmin + ((bounding_box.xmax - bounding_box.xmin)/2)), (bounding_box.ymin + ((bounding_box.ymax - bounding_box.ymin)/2)) 
        self.leftlow = (int(bounding_box.xmin),int(bounding_box.ymax)) #Readings are all taken from the 6 markers of the bounding box, not the button depth
        self.righthigh = (int(bounding_box.xmax),int(bounding_box.ymin))
        self.rightlow = (int(bounding_box.xmax ),int(bounding_box.ymax))
        self.lefthigh = (int(bounding_box.xmin),int(bounding_box.ymin))

    def converge(self,i):
        self.leftlow = (self.leftlow[0]+i,self.leftlow[1]-i)
        self.rightlow = (self.rightlow[0]-i,self.rightlow[1]-i)
        self.lefthigh = (self.lefthigh[0]+i,self.lefthigh[1]+i)
        self.righthigh = (self.righthigh[0]-i,self.righthigh[1]+i) 

if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))
    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov5Detector()
    #rospy.Subscriber('depth_to_rgb/image_raw', Image, callback)
    rospy.spin()
