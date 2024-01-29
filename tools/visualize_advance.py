from collections import defaultdict
from glob import glob
import os
import cv2
from tqdm import tqdm

def get_color(i):
    return [(i * 23 * j + 43) % 255 for j in range(3)]

def process(trk_path, img_list, output="output.mp4"):
    h, w, _ = cv2.imread(img_list[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 적절한 코덱으로 변경 가능
    out = cv2.VideoWriter(output, fourcc, 20.0, (w, h))

    tracklets = defaultdict(list)
    with open(trk_path, 'r') as file:
        for line in file:
            t, id, *xywhs = line.split(',')[:7]
            t, id = map(int, (t, id))
            x, y, w, h, s = map(float, xywhs)
            tracklets[t].append((id, *map(int, (x, y, x+w, y+h))))

    for i, path in enumerate(tqdm(sorted(img_list))):
        im = cv2.imread(path)
        for j, x1, y1, x2, y2 in tracklets[i + 1]:
            im = cv2.rectangle(im, (x1, y1), (x2, y2), get_color(j), 4)
            im = cv2.putText(im, f"{j}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, get_color(j), 2)
        out.write(im)

    out.release()

if __name__ == '__main__':
    trk_path = "../tracker/dancetrack0003.txt"
    img_list = glob("../data/Dataset/mot/DanceTrack/test/dancetrack0003/img1/*.jpg")
    output_path = '/data/MOTRv2/output/dancetrack0003.mp4'
    process(trk_path, img_list, output_path)
    print(f"비디오가 다음 경로에 저장되었습니다: {output_path}")
