from typing import List
import cv2
import torch
import folder_paths
import os
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path




class LoadCheckerboardImageForCalibrateCamera:

    @classmethod
    def INPUT_TYPES(cls):
        
        input_dir = folder_paths.get_input_directory()
        directories = []
        for item in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, item)) and item != "clipspace":
                directories.append(item)
        return {
            "required": {
                "directory": (directories,),
                "rows":("INT",{
                    "default": 3, 
                    "min": 3, #Minimum value
                    "max": 20, #Maximum value
                    "step": 1, #Slider's step
                    "display": "slider" # Cosmetic only: display as "number" or "slider"
                }),
                "cols":("INT",{
                    "default": 3, 
                    "min": 3, #Minimum value
                    "max": 20, #Maximum value
                    "step": 1, #Slider's step
                    "display": "slider" # Cosmetic only: display as "number" or "slider"
                }),
                
            }
        }
    RETURN_TYPES = ("MATRIX","DIST_COEF","RVECS","TVECS","INT","INT")
    RETURN_NAMES = ("matrix","dist_coef","rvecs","tvecs","h","w")# 内参矩阵，畸变系数，旋转量，偏移量
    FUNCTION = "calibrate"

    CATEGORY = "IG"
    def calibrate(self, directory: str,rows:int,cols:int, **kwargs):
        h = rows - 1
        w = cols - 1
        # 找棋盘格角点
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
        directory = folder_paths.get_annotated_filepath(directory.strip())
        dirs = Path(directory)
        objp = np.zeros((h*w,3), np.float32)
        objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        # objp = objp*grid_width * 10 
        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = [] # 在世界坐标系中的三维点
        imgpoints = [] # 在图像平面的二维点
        i=0
        for fname in dirs.iterdir():
            if fname.is_dir():
                continue
            # 只支持jpeg、jpg、png三种格式
            if fname.suffix in [".jpeg", ".jpg", ".png"]:
                img = cv2.imread(str(fname))
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                u, v = img.shape[:2]
                # 找到棋盘格角点
                ret, corners = cv2.findChessboardCorners(gray, (w,h),None)

                # cv2.drawChessboardCorners(img, (w, h), corners, ret)
                # cv2.imwrite('conimg'+str(i)+'.jpg', img)
                # 如果找到足够点对，将其存储起来
                if ret == True:
                    i+=1
                    objpoints.append(objp)
                    # 在原角点的基础上寻找亚像素角点
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    if [corners2]:
                        imgpoints.append(corners2)
                    else:
                        imgpoints.append(corners)
                    #追加进入世界三维点和平面二维点中
                    
                    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        return (
            mtx,
            dist,
            rvecs,
            tvecs,
            u,
            v
        )


class MatrixAndDistCoefToText:
    """
    去除畸变矫正
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "matrix": ("MATRIX",),
                "dist_coef": ("DIST_COEF",),
            }
            
        }
    
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("matrix_text","dist_coef_text")
    FUNCTION = "run"
    CATEGORY = "IG"

    def run(self,matrix,dist_coef,**kwargs):
        return (
            str(matrix),
            str(dist_coef)
        )

class Undistort:
    """
    去除畸变矫正
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "image":("IMAGE",),
                "matrix": ("MATRIX",),
                "dist_coef": ("DIST_COEF",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "IG"

    def run(self,image,matrix,dist_coef,**kwargs):
        _, h1, w1, _ = image.shape

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, dist_coef, (h1,w1), 0, (h1,w1))

        frame = image.squeeze(0).cpu().numpy()
        frame *= 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        dst1 = cv2.undistort(frame, matrix, dist_coef, None, newcameramtx)
        # # mapx,mapy=cv2.initUndistortRectifyMap(matrix,matrix,None,cameramtx,(w1,h1),5)
        # # dst2=cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
        x, y, w1, h1 = roi
        dst2 = dst1[y:y + h1, x:x + w1]
        img_rgb = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        return (
            torch.from_numpy(img_normalized).unsqueeze(0),
        )

NODE_CLASS_MAPPINGS = {
    "IG_LoadCheckerboardImageForCalibrateCamera":LoadCheckerboardImageForCalibrateCamera,
    "IG_MatrixAndDistCoefToText":MatrixAndDistCoefToText,
    "IG_Undistort":Undistort,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IG_LoadCheckerboardImageForCalibrateCamera":"Load Checkerboard Images for Calibrate Camera",
    "IG_MatrixAndDistCoefToText":"Matrix and distortion coefficient to text",
    "IG_Undistort":"Undistort",
}

if __name__ == "__main__":
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    # 图片路径
    image_path = '/mnt/sdb/ComfyUI/input/0156895594c6da32f87598b51f4784.jpg@1280w_1l_2o_100sh.jpg'

    # 使用PIL加载图片
    image = Image.open(image_path).convert("RGB")

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    # 应用转换器
    print(image.shape)
    _,h1, w1, _ = image.shape
    frame = image.squeeze(0).cpu().numpy()
    frame *= 255
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img_normalized = img_rgb.astype(np.float32) / 255.0
    print(frame.shape)
    cv2.imwrite("test.jpg",frame)