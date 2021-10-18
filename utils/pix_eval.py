import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
import glob as gb


videos = [3, 4, 14, 18, 19, 21, 22, 23, 24, 32]
videos = list(range(1, 37))

gt_base_path = '/home/sensifai/behnam/anomaly/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/'
gt_base_path = '/home/sensifai/behnam/anomaly/pix2pix/video-anomaly-gan/results/UCSD_Ped1/'
pred_base_path = '/home/sensifai/behnam/anomaly/pix2pix/video-anomaly-gan/results/gen_out/raw_alt_wob_42/'
pred_base_path = '/home/sensifai/behnam/anomaly/pix2pix/video-anomaly-gan/results/gen_out/16frames_raw_wob16/'


clip_pixel_count = 199*256*256
img_pixel_count = 158*238

pred_all = np.zeros((len(videos)*clip_pixel_count))
gt_all = np.zeros((len(videos)*clip_pixel_count))
auc_list = []
for i, v in enumerate(videos):
    gt_path = gt_base_path + 'Test{:03d}_gt/*.bmp'.format(v)
    pred_path = pred_base_path + '{:03d}/*.jpg'.format(v-1)
    gt_images_path = gb.glob(gt_path)
    pred_images_path = gb.glob(pred_path)
    gt_images_path.sort()
    pred_images_path.sort()
    for j, (gt_path, pred_path) in enumerate(zip(gt_images_path[1:], pred_images_path)):
        gt_img = cv2.imread(gt_path, 0).astype('float32')
        # gt_img = cv2.resize(gt_img, (256, 256))
        if gt_img.max() > 0:
            gt_img = gt_img / gt_img.max()
        
        pred_img = cv2.imread(pred_path, 0).astype('float32')
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

        if pred_img.max() > 0:
            pred_img = pred_img / pred_img.max()

        gt = gt_img.ravel()
        pred = pred_img.ravel()
        pred_all[i*clip_pixel_count+j*img_pixel_count:i*clip_pixel_count+(j+1)*img_pixel_count] = pred
        gt_all[i*clip_pixel_count+j*img_pixel_count:i*clip_pixel_count+(j+1)*img_pixel_count] = gt
    auc = roc_auc_score(gt_all[i*clip_pixel_count:(i+1)*clip_pixel_count], pred_all[i*clip_pixel_count:(i+1)*clip_pixel_count])
    auc_list.append(auc)
    print(i, auc)

auc_avg = np.mean(auc_list)
print(auc_avg)
# auc = roc_auc_score(gt_all, pred_all)
# print('overall:', auc)







anomalies = [
    [60,152]
,[50,175]
,[91,200]
,[31,168]
,[5,90, 140,200]
,[1,100, 110,200]
,[1,175], [1,94], [1,48], [1,140],
[70,165], [130,200], [1,156], [2,200], [138,200], [123,200], [1,47], [54,120], [64,138], [45,175], [31,200], [16,107], [8,165], [50,171]
,[40,135], [77,144], [10,122], [105,200], [1,15, 45,113],[ 175,200], [1,180], [1, 52, 65, 115], [5,165], [1,121], [86,200], [15,108]
]
print(len(anomalies))
preds = np.zeros(200*len(videos))
gt = np.zeros_like(preds)
auc_arr = []
k = np.ones((3, 3), 'uint8')
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
disc = False
for i, v in enumerate(videos):
    files_dir = '../results/gen_out/16frames_raw_wob16/{:03d}/*jpg'.format(v-1)
    files_path = gb.glob(files_dir)
    files_path.sort()
    an = anomalies[v-1]
    for j, fp in enumerate(files_path):
        img = cv2.imread(fp, 0).astype('float32')
        img = img ** 5

        # ret, thr = cv2.threshold(img,20,255,cv2.THRESH_TOZERO)
        # img = cv2.erode(img, k).sum()
        preds[i*200+j+1] = img.sum()

    gt[i*200+an[0]-1:i*200+an[1]] = 1
    if len(an) == 4:
        gt[i*200+an[2]-1:i*200+an[3]] = 1
    preds[i*200:] /= preds[i*200:].max()
    if disc:
        preds[i*200:] = np.abs(preds[i*200:] - 1)
    auc = roc_auc_score(gt[200*i:200*(i+1)], preds[200*i:200*(i+1)])
    auc_arr.append(auc)
    print(i, auc)
print(np.mean(auc_arr))

preds = np.array(preds)
preds_normal = preds / preds.max()





for i, an in enumerate(anomalies):
    
    gt[i*200+an[0]-1:i*200+an[1]] = 1
    if len(an) == 4:
        gt[i*200+an[2]-1:i*200+an[3]] = 1

auc = roc_auc_score(gt, preds_normal)
print(auc)