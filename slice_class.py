# -*- coding: utf-8 -*-
"""
Time    : 2022/4/25 13:02
Author  : cong
"""

import threading


class MyThread(threading.Thread):
    def __init__(self, front_frames_action, out_writer):
        threading.Thread.__init__(self)
        self.front_frames_action = front_frames_action
        self.out_writer = out_writer
        print('action thread ')

    def run(self):
        for i in self.front_frames_action:
            self.out_writer.write(i)
        self.out_writer.release()

