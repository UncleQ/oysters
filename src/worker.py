# coding=utf-8
import threading


class Worker (threading.Thread):   #继承父类threading.Thread
    def __init__(self, thread_id, name, counter):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.counter = counter

    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        print("Starting " + self.name)
        #print_time(self.name, self.counter, 5)
        print("Exiting " + self.name)