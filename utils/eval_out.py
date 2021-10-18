import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
import glob as gb


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
preds = np.zeros(200*36)
gt = np.zeros_like(preds)
auc_arr = []
k = np.ones((3, 3), 'uint8')
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
disc = False
for i in range(36):
    files_dir = '../results/gen_out/16frames_raw_wob16/{:03d}/*jpg'.format(i)
    files_path = gb.glob(files_dir)
    files_path.sort()
    an = anomalies[i]
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