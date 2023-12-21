
from concurrent.futures import ThreadPoolExecutor
import cv2
import onnxruntime as rt
import numpy as np
import os
from re import sub
import datetime
from pathlib import Path
import shutil
def times(contador, fps):
    td = datetime.timedelta(seconds=(int(contador)/fps))
    ms = td.microseconds // 1000
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return '{:02d}_{:02d}_{:02d}_{:03d}'.format(h, m, s, ms)

provider = []
for providerElement in rt.get_available_providers():
     if( providerElement == 'DmlExecutionProvider'):
         provider.append('DmlExecutionProvider')
if(len(provider)==0):
    provider.append('CPUExecutionProvider')
sess = rt.InferenceSession('end2end.onnx', providers=provider)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
executor = ThreadPoolExecutor(max_workers=4)
executor2 = ThreadPoolExecutor(max_workers=1)
def guardarImagen(directory,videofile,oldcounter,counter,fps):
    # print(f'{times(oldcounter,fps)}__{times(counter,fps)}.jpeg')
    # print()
    cv2.imwrite(os.path.join(directory,f'{times(oldcounter,fps)}__{times(counter,fps)}.jpeg'), videofile)


def filtradoNombre(nombre):
    x = sub(r"\.avi$", "", nombre)
    x = sub(r"\.mp4$", "", x)
    x = sub(r"\.mkv$", "", x)
    x = sub(r"\.ts$", "", x)
    x = sub(r"\.rmvb", "", x)
    x = x.strip()
    x   = ''.join(e for e in x if e.isalnum())
    return x
def runOCR(direcoryFrames):
    os.system(f'cd {direcoryFrames} && python ../ocrGoogle.py')
    shutil.rmtree(f'/{direcoryFrames}')
def imgExtractor(videofile,framesDir):
    video = cv2.VideoCapture(str(videofile))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps)
    directory = f'{os.getcwd()}/{framesDir}'
    counter = 1
    nFramePrevius=0
    nFrameNow = 0
    oldFrame=[]
    newFrame=[]
    global h 
    global w
    mask_temp = []
    oldContainText =False
    containText =False
    imgChange =False
    while counter <= total_frames:
        _, img = video.read()
        if not _ : 
            break
        if counter ==1:
            img = cv2.resize(img,(int(img.shape[1]*(480/img.shape[0])),480), interpolation=cv2.INTER_AREA)
            img = img[int(img.shape[0]/1.3):, :, :]
            h=img.shape[0]
            w=img.shape[1]
        if counter % int(fps/10) == 0:
            containText = False
            imgChange = False
            img = cv2.resize(img,(int(img.shape[1]*(480/img.shape[0])),480), interpolation=cv2.INTER_NEAREST)
            img = img[int(img.shape[0]/1.3):, :, :]
            img2 = img.copy()
            img_detection = img.astype(np.float32)/255
            img_detection = img_detection.transpose([2,0,1])

            img_detection = np.expand_dims(img_detection, axis=0)
            images = np.vstack([img_detection])
            auxtensor = rt.OrtValue.ortvalue_from_numpy(images)
            mask = sess.run([label_name], {input_name: auxtensor})[0][0]
            mask = (mask*255).astype(np.uint8)
            
            mask = cv2.bitwise_not(mask)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
            mask = cv2.erode(mask, np.ones((3, 7)), iterations=2)
            if len(mask_temp)==0:
                mask_temp = mask
            contours =  cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
            for c in contours:
                area = cv2.contourArea(c)
                if 300 <area < w*h/2:
                    cv2.drawContours(img, [c], -1, (0,255,0), 1)
                    containText = True

            res2 = cv2.absdiff(mask_temp, mask)
            res2 = cv2.bitwise_not(res2)
            # cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8),iterations=3)
            res2 = cv2.dilate(res2, np.ones((6, 15)), iterations=2)
            contours =  cv2.findContours(res2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]

            for c in contours:
                area2 = cv2.contourArea(c)
                if 100 <area2 < w*h/2:
                    cv2.drawContours(img, [c], -1, (0,0,255), 4)
                    imgChange = True
            # print(imgChange,oldContainText,containText)
            # nFrameNow = counter
            if imgChange & oldContainText:
                if len(oldFrame)==0:
                    oldFrame = img2
                executor.submit(guardarImagen,directory,oldFrame,nFramePrevius,counter,fps)
                # guardarImagen(oldFrame,nFramePrevius,counter-3,fps)
            if imgChange & containText:
                oldFrame = img2
                nFramePrevius= counter

            # if imgChange==False & containText:
            #     nFramePrevius = counter
            

            # cv2.imshow("test2", res2)
            mask_temp = mask
            oldContainText = containText
            # cv2.imshow("entrada", img)
            # cv2.imshow("test", mask)
            # cv2.waitKey(1)
        counter += 1
    
# imgExtractor('video\SORTIE! Machine Robo Rescue Ep 1 ( AI HD Restoration project).mp4')
current_directory = Path(Path.cwd())
for video in os.scandir('videos'):
    # imgExtractor(video.path)
    direcoryFrames = filtradoNombre(video.name)
    # print(Path(f'{current_directory}/{direcoryFrames}'))
    if not Path(f'{current_directory}/{direcoryFrames}').exists() :
        os.mkdir(direcoryFrames)
        print('se creo directorio')
        imgExtractor(video.path,direcoryFrames)
    executor2.submit(runOCR,direcoryFrames)

executor.shutdown(wait=True)
executor2.shutdown(wait=True)


