# -*- coding: utf-8 -*-
"""
Time    : 2022/5/3 08:15
Author  : cong
"""
from slice_class import MyThread
from PIL import Image
from yolo_slim import *
import copy, math, cv2, datetime
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import threading
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
yolo = YOLO()

#检测模型的配置参数
#rtsp
# camera_pos, camera_rtsp, camera_radius = get_camera("10.31.97.16",radius = 300)
# video_path = camera_rtsp
# radius = camera_radius

# -------------------------------------------------------------------------#
#local
video_path      = "111.mp4"
video_save_path = "222.mp4"
log_name        =  "event_info.npy"
radius = 310
interval = 3
threshold_hatch = 900 / interval
# -------------------------------------------------------------------------#

video_fps = 25
capture = cv2.VideoCapture(video_path)
fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
total_frame = capture.get(7)
print('total_frame:', total_frame)
ref, frame = capture.read()
if not ref:
    raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

#nodes 的定义列表
# -------------------------------------------------------------------------#
nodes = ["hatch_open",
         "hatch_close",
         "unload_cargo",
         "load_cargo", ]

# 记录触发事件（开关舱门、装卸货）的类型及时间
event_info, text_3, info_hatch_10= [],[],[]
#参数初始化
hatch_dic,cargo_dic ,del_hatch_dic,pre_hatch_cargo_match,cargo_info,hatch_info = {},{},{},{},{},{}
hatch_num,hatch_id,c_num,c_id,count_50,count,fps=[0,0,0,0,0,0,0]


# 添加切片所需参数
count_id = 0
front_frames = []
action_frames = [-51*3]
action = False

event_infos = [[]]
while ref:
    t_o = time.time()
    ref, frame = capture.read()
    if not ref:
        break
    count+=1
    count_id += 1
    if count % interval == 0:
        # -------------------------------------------------------------------------#
        count = 0
        count_50 += 1
        # -------------------------------------------------------------------------#
        #图像进行处理并检测
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pic_name = str(datetime.datetime.now()).split(" ")[0] + "_" + "_".join(
            str(datetime.datetime.now()).split(" ")[-1].split(":"))
        # cv2.imwrite("logs/image/%s.jpg" % pic_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        frame = Image.fromarray(np.uint8(frame))
        hatch, cargo = yolo.detect_image(frame)
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, "frame_id: %d " % (count_id),  (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        info_hatch_10.append([])
        # -------------------------------------------------------------------------#
        #  视频切片前20帧
        print('count_id:', count_id)
        front_frames.append(frame)
        if len(front_frames) > 100:
            front_frames = front_frames[-50:]
        # -------------------------------------------------------------------------#
        # 进行逻辑判断
        hatchatch_id = {}
        if hatch: #hatch坐标信息
        # -------------------------------------------------------------------------#
            #画出舱门的ROI
            for hatch_box in hatch:
                top, left, bottom, right = hatch_box
                hatch_centor = (int(left + (right - left) / 2), int(top + (bottom - top) / 2))
                frame_roi = cv2.circle(frame, hatch_centor, radius, (255, 0, 0), 2, 8, 0)
                pic_name = str(datetime.datetime.now()).split(" ")[0] + "_" + "_".join(
                str(datetime.datetime.now()).split(" ")[-1].split(":"))
            cv2.imwrite("logs/detect_image/%s.jpg" % pic_name, frame_roi, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            hatch_num += 1
            if hatch_num == 1:
                for h in hatch:
                    hatch_id += 1
                    hatch_dic[hatch_id] = [int(((h[1] + h[3]) / 2)), int(((h[0] + h[2]) / 2))]
                    event_info.append("hatch_open:" + str(hatch_id) + " time:" + str(datetime.datetime.now()))
                    hatch_info[hatch_id] = "hatch_open"
                    cargo_info[hatch_id] = []
                    hatchatch_id[hatch_id] = h
                    info_hatch_10[-1].append(hatch_id)
                    save_print_logs(log_name, event_info)

            else:
                hatch_copy = copy.deepcopy(hatch)
                min_id = {}
                for i in hatch_dic:
                    x, y = hatch_dic[i][0], hatch_dic[i][1]
                    tmp = []
                    for h in hatch:
                        dis = math.sqrt(
                            pow(x - int(((h[1] + h[3]) / 2)), 2) + pow(y - int(((h[0] + h[2]) / 2)), 2))
                        tmp.append(dis)
                    # print(str(num) + '_' + 'tmp:', tmp)
                    # 2帧舱门间的的距离小于80，才化为同一舱门
                    if min(tmp) <= 80:
                        index = tmp.index(min(tmp))
                        min_id.setdefault(index, {})[i] = min(tmp)
                # print('min_id:', min_id) #
                for i in min_id:
                    min_count_key = min(min_id[i], key=min_id[i].get)
                    hatch_dic[min_count_key] = [int(((hatch[i][1] + hatch[i][3]) / 2)),
                                                int(((hatch[i][0] + hatch[i][2]) / 2))]
                    hatch_copy.remove(hatch[i])
                    hatchatch_id[min_count_key] = hatch[i]
                    info_hatch_10[-1].append(min_count_key)
                # 与删除的舱门信息对比，避免同一舱门的id不一致情况
                min_id_del = {}
                for i in del_hatch_dic:
                    x, y = del_hatch_dic[i][0], del_hatch_dic[i][1]
                    tmp_del = []
                    for h in hatch:
                        dis = math.sqrt(
                            pow(x - int(((h[1] + h[3]) / 2)), 2) + pow(y - int(((h[0] + h[2]) / 2)), 2))
                        tmp_del.append(dis)
                    # 2帧舱门间的的距离小于100，才化为同一舱门
                    if min(tmp_del) <= 100:
                        index = tmp_del.index(min(tmp_del))
                        min_id_del.setdefault(index, {})[i] = min(tmp_del)
                # print('min_id_del:', min_id_del)
                for i in min_id_del:
                    # 防止当前舱门存在与前一帧检测的舱门对应1个id，又与删掉的舱门对应同一id的情况发生
                    if hatch[i] in hatch_copy:
                        min_count_key = min(min_id_del[i], key=min_id_del[i].get)
                        hatch_dic[min_count_key] = [int(((hatch[i][1] + hatch[i][3]) / 2)),
                                                        int(((hatch[i][0] + hatch[i][2]) / 2))]
                        hatch_copy.remove(hatch[i])
                        hatchatch_id[min_count_key] = hatch[i]
                        info_hatch_10[-1].append(min_count_key)
                        cargo_info[min_count_key] = []
                        hatch_info[min_count_key] = "hatch_open"
                        del del_hatch_dic[min_count_key]
                        event_info.append("hatch_open:" + str(min_count_key) + " time:" + str(datetime.datetime.now()))
                        text_3.append(['hatch_open', str(hatch_id), datetime.datetime.now()])
                        if len(text_3) > 3:
                            text_3.pop(0)
                            event_info = solve_exception(text_3, event_info)
                        save_print_logs(log_name, event_info)


                for h in hatch_copy:
                    hatch_id += 1
                    hatch_dic[hatch_id] = [int(((h[1] + h[3]) / 2)), int(((h[0] + h[2]) / 2))]
                    event_info.append("hatch_open:" + str(hatch_id) + " time:" + str(datetime.datetime.now()))
                    text_3.append(['hatch_open',str(hatch_id),datetime.datetime.now()])
                    if len(text_3)>3:
                        text_3.pop(0)
                        event_info=solve_exception(text_3, event_info)
                    save_print_logs(log_name, event_info)
                    hatchatch_id[hatch_id] = h
                    info_hatch_10[-1].append(hatch_id)
                    cargo_info[hatch_id] = []
                    hatch_info[hatch_id] = "hatch_open"


        if len(info_hatch_10) > threshold_hatch:
            info_hatch_10.pop(0)
        info = [n for a in info_hatch_10 for n in a]
        hatch_dic_copy = copy.deepcopy(hatch_dic)
        for i in hatch_dic_copy:
            if i not in info:
                event_info.append("hatch_close:" + str(i) + " time:" + str(datetime.datetime.now()))
                text_3.append(['hatch_close', str(hatch_id), datetime.datetime.now()])
                if len(text_3) > 3:
                    text_3.pop(0)
                    event_info = solve_exception(text_3, event_info)
                save_print_logs(log_name, event_info)
                hatch_info[i] = "hatch_close"
                # 避免舱门close后还出现装卸货信息
                if i in cargo_info:
                    cargo_info[i] = []
                # 记录删除的舱门id及中心位置坐标
                del_hatch_dic[i] = hatch_dic[i]
                del hatch_dic[i]

        # -------------------------------------------------------------------------#
        # 2.装、卸货检测
        hatch_cargo_match = {}
        if cargo and hatchatch_id:
            for h in hatchatch_id:
                x, y = int(((hatchatch_id[h][1] + hatchatch_id[h][3]) / 2)), int(
                    ((hatchatch_id[h][0] + hatchatch_id[h][2]) / 2))
                c_roi = {}
                for c in cargo:
                    x1, y1 = int(((c[1] + c[3]) / 2)), int(((c[0] + c[2]) / 2))
                    dis = math.sqrt(pow(x - x1, 2) + pow(y - y1, 2))
                    # 划定roi区域：与对应的舱门中心坐标距离小于150
                    if dis <= radius and y1 > y:
                        c_roi[tuple(c)] = dis
                if c_roi:
                    cargo_1 = min(c_roi, key=c_roi.get)
                    hatch_cargo_match[h] = [int(((cargo_1[1] + cargo_1[3]) / 2)),
                                            int(((cargo_1[0] + cargo_1[2]) / 2))]
        # print('pre_cur:', count, count_50, pre_hatch_cargo_match, hatch_cargo_match)
        if count_50 == 16:
            count_50 = 0
            if pre_hatch_cargo_match and hatch_cargo_match:
                for i in hatch_cargo_match:
                    if i in pre_hatch_cargo_match:
                        dis = abs(math.sqrt(pow(pre_hatch_cargo_match[i][0] - hatch_cargo_match[i][0], 2) +
                                            pow(pre_hatch_cargo_match[i][1] - hatch_cargo_match[i][1], 2)))
                        print("dis:", dis)
                        if dis <= 30:
                            if hatch_cargo_match[i][1] - pre_hatch_cargo_match[i][1] >= 3:
                                if not cargo_info[i] or cargo_info[i][-1] != "unload_cargo":
                                    cargo_info[i].append("unload_cargo")
                                    event_info.append(
                                        "unload_cargo:" + str(i) + " time:" + str(datetime.datetime.now()))
                                    text_3.append(['unload_cargo', str(i), datetime.datetime.now()])
                                    if len(text_3) > 3:
                                        text_3.pop(0)
                                        event_info = solve_exception(text_3, event_info)
                                    save_print_logs(log_name, event_info)
                            if pre_hatch_cargo_match[i][1] - hatch_cargo_match[i][1] >= 3:
                                if not cargo_info[i] or cargo_info[i][-1] != "load_cargo":
                                    cargo_info[i].append("load_cargo")
                                    event_info.append("load_cargo:" + str(i) + " time:" + str(datetime.datetime.now()))
                                    text_3.append(['load_cargo', str(i), datetime.datetime.now()])

                                    if len(text_3) > 3:
                                        text_3.pop(0)
                                        event_info = solve_exception(text_3, event_info)
                                    save_print_logs(log_name, event_info)
            pre_hatch_cargo_match = hatch_cargo_match

        # -------------------------------------------------------------------------#
        # 添加切片代码
        # 判断第一个从0开始查找的切片
        if event_info:
            print('event_info:', event_info)
            print('event_infos[-1]:', event_infos[-1])
            frame_id_compare = count_id - action_frames[-1]
            print('frame_id_compare:', frame_id_compare)
            if event_info != event_infos[-1] and frame_id_compare > 51*interval:
                action = True
                after_frames_action = []
                front_frames_action = front_frames[-50:-1]
                action_frame = count_id
                print('action_frame:', action_frame)
                action_frames.append(action_frame)
                event_infos.append(copy.deepcopy(event_info))
                save_path = [event_infos][-1][-1][-1] + '.mp4'
                print('save_path:', save_path)
                out_writer = cv2.VideoWriter(save_path, fourcc, video_fps, size)

        # # 切片操作
        if action:
            after_frames_action.append(frame)
            print('len(front_frames_action):', len(front_frames_action))
            print('len(after_frames_action):', len(after_frames_action))
            if len(after_frames_action) == 51:
                if len(front_frames_action) == 0:
                    frames_action = after_frames_action
                else:
                    frames_action = np.concatenate((front_frames_action, after_frames_action))
                print('len(frames_action):', len(frames_action))

                thread1 = MyThread(frames_action, out_writer)
                thread1.start()
                print('current_thread_name:', thread1.getName())
            if len(after_frames_action) > 51:
                action = False

        length = len(threading.enumerate())
        print('当前运行的线程数为：%d' % length)
        # if length == 2:
        #     action = False


        #-------------------------------------------------------------------------#
        #每次处理一帧图像后输出的状态信息
        # print("event_info %s;len :%s"%(event_info,len(event_info)))
        # print("text_3 %s;len :%s"%(text_3,len(text_3)))
        # print("info_hatch_10 %s;len :%s"%(info_hatch_10,len(info_hatch_10)))
        # -------------------------------------------------------------------------#
        # 3.保存、记录信息
        fps = (fps + (1. / (time.time() - t_o))) / 2
        print("fps= %.2f" % (fps))

        if video_save_path != "":
            out.write(frame)


print("Video Detection Done!")
capture.release()
if video_save_path != "":
    print("Save processed video to the path :" + video_save_path)
    out.release()
cv2.destroyAllWindows()
yolo.close_session()

