from collections import defaultdict
from glob import glob
import json
import os
import cv2
from tqdm import tqdm


def get_color(i):
    return [(i * 23 * j + 43) % 255 for j in range(3)]

with open("/data/MOTRv2/data/Dataset/mot/det_db_motrv2.json") as f:
    det_db = json.load(f)


def process(trk_path, img_list, output="output.mp4"):
    h, w, _ = cv2.imread(img_list[0]).shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec according to your preference
    out = cv2.VideoWriter(output, fourcc, 20.0, (w, h))

    tracklets = defaultdict(list)
    for line in open(trk_path):
        t, id, *xywhs = line.split(',')[:7]
        t, id = map(int, (t, id))
        x, y, w, h, s = map(float, xywhs)
        tracklets[t].append((id, *map(int, (x, y, x+w, y+h))))

    for i, path in enumerate(tqdm(sorted(img_list))):
        im = cv2.imread(path)
        for det in det_db[path.replace('.jpg', '.txt')]:
            x1, y1, w, h, _ = map(int, map(float, det.strip().split(',')))
            im = cv2.rectangle(im, (x1, y1), (x1+w, y1+h), (255, 255, 255), 6)
        for j, x1, y1, x2, y2 in tracklets[i + 1]:
            im = cv2.rectangle(im, (x1, y1), (x2, y2), get_color(j), 4)
            im = cv2.putText(im, f"{j}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, get_color(j), 2)
        out.write(im)

    out.release()


if __name__ == '__main__':
    jobs = os.listdir("../tracker/")
    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    jobs = sorted(jobs)[rank::ws]
    for seq in jobs:
        print(seq)

        trk_path = "../tracker/dancetrack0003.txt"

        img_list = glob(f"../data/Dataset/mot/DanceTrack/test/dancetrack0003/img1/*.jpg")
        process(trk_path, img_list, f'motr_trainval_demo/dancetrack0003.mp4')
        break
