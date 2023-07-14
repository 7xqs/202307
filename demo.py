from rknn.api import RKNN
#import imutils
import os
#import urllib
#import traceback
import time
import sys
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont #打印中文字时候需要
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


VIDEO_PATH = './5.mp4'
RKNN_MODEL = './5s.rknn'
BOX_THRESH = 0.4
SCORE_THRESH = 0.6
NMS_THRESH = 0.3 #0.6 
IMG_SIZE = 640

CLASSES = ("person", "bicycle", "car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
           "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
           "bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors","teddy bear","hair drier", "toothbrush")

id_list = []
number = 0
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def plot_bboxes(image, bboxes, line_thickness=None):
    global number
    global id_list
    is_red = True #红灯信号开启闯红灯检测
    
    im_size = image.shape
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    
    #print(bboxes)
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if CLASSES[cls_id] in ['person']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(CLASSES[cls_id], 0, fontScale=tl / 3, thickness=tf)[0]
        
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        
        #print(t_size,c2)
        
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled       
        cv2.putText(image, '{} ID-{}'.format(CLASSES[cls_id], pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)       
        
        if (not pos_id in id_list) and (y2 < im_size[0]*0.65) and (y2 > im_size[0]*0.63) and is_red:      #检测行人闯红灯
            id_list.append(pos_id)
            number+=1
            print(id_list)
        
        
    #cv2.putText(image, 'Count:{}'.format(number), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)    
    #cv2.putText(image, 'ID-list:{}'.format(id_list), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    #如果希望图中显示中文需要对图片进行转换
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)  
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg) # 图片上打印
    font = ImageFont.truetype("simhei.ttf", 50, encoding="utf-8") # 参数1：字体文件路径，参数2：字体大小
    draw.text((10, 30), '车流量：{}'.format(number), (255, 0, 0), font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    draw.text((10, 90), '通过车辆ID：{}'.format(id_list), (255, 0, 0), font=font) # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    #draw.text((10, 90), '置信度：' + str(prediction), (255, 0, 0), font=font)
    #draw.text((10, 90), '检测用时：{:5f}秒'.format(t2-t1) , (255, 0, 0), font=font)
    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

    return cv2charimg  #image



def update_tracker(pred_boxes, image):
    """
    轨迹追踪
    :param target_detector: 检测器
    :param image: 需要检测的图像
    :return: 经过plot_bboxes函数绘制后的图像
    """
    bbox_xywh = []
    confs = []
    bboxes2draw = []
    lbls = []
    if len(pred_boxes):
        for x1, y1, x2, y2, lbl, conf in pred_boxes:
            obj = [
                int((x1+x2)/2), int((y1+y2)/2),
                x2-x1, y2-y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)
            lbls.append(lbl)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)
        lblss = torch.Tensor(lbls)

        # Pass detections to deepsort
        outputs = deepsort.update(xywhs, confss, lblss, image)

        
        for value in list(outputs):
        
            x1,y1,x2,y2,track_id,class_id = value
            #print(lbls,class_id)
            bboxes2draw.append(
                (x1, y1, x2, y2, class_id, track_id)     
            )
            


    image = plot_bboxes(image, bboxes2draw)

    return image   #, new_faces, face_bboxes


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_classes = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_confidences[...,0] >= BOX_THRESH)


    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
              [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes, dw, dh ,r):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    #s = ''
    
    pred_boxes = []
    im_size = image.shape        
    print(im_size)        
    for box, score, cl in zip(boxes, scores, classes):
                        
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        top = (top-dw)/r
        left = (left-dh)/r
        right = (right-dw)/r
        bottom = (bottom-dh)/r        
        print('box coordinate left,top,right,down: [{},{},{},{}]'.format(top, left, right, bottom))

               
        x1 = int(top)
        y1 = int(left)
        x2 = int(right)
        y2 = int(bottom)
        lbl = CLASSES[cl]
        
        if lbl in ['car','bus','truck'] and score >= SCORE_THRESH:
            #pred_boxes.append((x1, y1, x2, y2, lbl, score))  
            #pred_boxes.append((x1, y1, x2, y2, cl, score))
            pred_boxes.append((x1, y1, x2, y2, 2, score))  #车流量统计，小车公交卡车都应该归为车辆
            
        if lbl in ['person', 'traffic light'] and score>0.5:       
            #cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            if (lbl=='person') and (x2 < im_size[1]*0.7) and (x1 > im_size[1]*0.3):
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, '!!!{0} {1:.2f}'.format(lbl, score),
                            (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(image, '{0} {1:.2f}'.format(lbl, score),
                            (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

        
    return image, pred_boxes


def letterbox(im, new_shape=(IMG_SIZE, IMG_SIZE), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio =  r   #r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



def load_rknn_model(PATH):
    rknn = RKNN()
    print('--> Loading model')
    ret = rknn.load_rknn(PATH)
    if ret != 0:
        print('load rknn model failed')
        exit(ret)
    print('done')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn
        
        
        
if __name__ == '__main__':


    #yolov5
    rknn = load_rknn_model(RKNN_MODEL)
    
    #deepsort
    cfg = get_config()
    cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")

    reid_model = './deep_sort/deep_sort/deep/checkpoint/ckpt.rknn'
    extractor = load_rknn_model(reid_model)
    deepsort = DeepSort(extractor,  # 传入RKNN对象
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_rknn=True)


    #fps = int(capture.get(5))
    fps = 10
    print('fps:', fps)
    t = int(1000 / fps)   
       
    cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN) 
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    skip =True #跳过检测步骤
    
    if not skip:
        capture = cv2.VideoCapture(VIDEO_PATH) 
        videoWriter = None  #save the video
        count=0
        ret, img = capture.read()    
        while(ret):
            # Set inputs
            #img = cv2.imread(IMG_PATH)
            img_1, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
     
     
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
            #img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
            # Inference
            print('--> Running model')
            time1=time.time()
            outputs = rknn.inference(inputs=[img_1])
            print(time.time()-time1)
            


            # post process
            input0_data = outputs[0]
            input1_data = outputs[1]
            input2_data = outputs[2]

            input0_data = input0_data.reshape([3,-1]+list(input0_data.shape[-2:]))
            input1_data = input1_data.reshape([3,-1]+list(input1_data.shape[-2:]))
            input2_data = input2_data.reshape([3,-1]+list(input2_data.shape[-2:]))

            input_data = list()
            input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

            boxes, classes, scores = yolov5_post_process(input_data)
            #print(boxes, classes, scores)
            #img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            pred_boxes = []
            if boxes is not None:
                img, pred_boxes = draw(img, boxes, scores, classes, dw, dh, ratio)      
                print(pred_boxes)
    
            time2=time.time()
            print('目标检测用时', time2-time1)

            time3=time.time()
            img = update_tracker(pred_boxes, img)                    
            time4=time.time()
            
            print('追踪用时', time4-time3)
            
            
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  
                videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (img.shape[1], img.shape[0]))
                
            videoWriter.write(img)                 
            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            count+=1

            #if count==3:
            #    break
            ret, img = capture.read()  
            
            #if cv2.getWindowProperty('Image', cv2.WND_PROP_AUTOSIZE) < 1:
            #    break
                    
        
        videoWriter.release()


    rknn.release()

    # 1.初始化读取视频对象
    cap = cv2.VideoCapture("./result_car3.mp4")

    # 2.循环读取图片
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Image', frame)
        else:
            print("视频播放完成！")
            break

        # 退出播放 
        if cv2.waitKey(int(1000/t)) == 27:  # 按键esc
            break

    # 3.释放资源
    cap.release()
    cv2.destroyAllWindows()
