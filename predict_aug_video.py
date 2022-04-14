
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from yolo import YOLO


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

if __name__ == "__main__":
    yolo = YOLO()
    video_path = 'first.avi'
    video_save_path = 'first_out.mp4'
    capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    ref, frame = capture.read()
    fps = 0.0
    count = 0
    while (True):
        t = time.time()
        ref, frame = capture.read()
        if not ref:
            break
        t1 = time.time()
        count += 1
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame1 = Image.fromarray(np.uint8(frame))
        ss = yolo.detect_image(frame1)
        # 进行检测
        frame = np.array(ss)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        print('current_frame:', count)
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
        out.write(frame)
        if c == 27:
            capture.release()
            break
    print("Video Detection Done!")
    capture.release()

    if video_save_path != "":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()
