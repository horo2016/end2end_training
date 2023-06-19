import cv2
import numpy as np
import argparse
import os
import sys
import datetime
import json
from paho.mqtt import client as mqtt_client
import random
import time ,threading
#MQTT相关
broker = '127.0.0.1'
#broker = 'www.woyilian.com'
port = 1883
topic = "/camera/collect"
cmd_vel_topic = "/cmd/vel"
client_id = 'python-mqtt-{}'.format(random.randint(0, 1000))



orgFrame = None
ret = False
Running = True
detect_ok = False

def Camera_isOpened():
    global  cap
        #开始捕获摄像头视频流
    cap = cv2.VideoCapture('/dev/video0')
try:
    Camera_isOpened()
       
except:
    print('Unable to detect camera! \n')
    
def get_image():
    global orgFrame
    global ret
    global Running
    global  cap
 
    while True:
        if Running:
            if cap.isOpened():
                ret, orgFrame = cap.read()
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.01)

th1 = threading.Thread(target = get_image)
th1.setDaemon(True)
th1.start()


#定义mqtt的client
client = mqtt_client.Client(client_id)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("test")

def on_message(client, userdata, msg):
    global orgFrame
    print(msg.topic+":"+msg.payload.decode("utf-8"))
    #_switch = ord(msg.payload.decode("utf-8"))-48
    country_dict = json.loads(msg.payload)
    print(country_dict["control"])
    print(country_dict["vel"])
    print(country_dict["ang"])
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%Y%m%d%H%M%S%f')
    with open("../dataset/driving_log.csv","a") as file:
        file.write(country_dict["vel"]+","+ country_dict["ang"]+","+time_str+".jpg"+"\n")        
    cv2.imwrite("../dataset/%s.jpg"%time_str, orgFrame) 


def publish(client):
    global last_time
    global detect_ok
    global orgFrame
    msg_count = 0
    while True:
        if detect_ok is True:
            detect_ok = False
            #重新裁剪大小
            #img = cv2.resize(img,(512,256))
            #if _switch == 0:#如果为0 直接重头开始
                #continue
            #cv2.imshow("IN", img)
            #是否写入保存的文件
            #if args.savevideo:
                #out.write(orgFrame)
            t=time.time()
            #print (int(round(t * 1000)))
            img_encode = cv2.imencode('.jpg', orgFrame)[1] #.jpg 编码格式
            #t=time.time()
            #print (int(round(t * 1000)))
            
            #curr_time = datetime.datetime.now()
            #time_str = datetime.datetime.strftime(curr_time,'%Y%m%d%H%M%S')
            
            #framecount += 1
            byteArr = bytearray(img_encode)#1xN维的数据
            #print(byteArr)
            #print(len(byteArr))
            #每1秒钟采集一次
            #cv2.imwrite("img/%s.jpg"%time_str, img) 
            time.sleep(0.001)
            if (t*1000 - last_time*1000 >= 500):
                result = client.publish(topic,byteArr,0)
                last_time = t
                # result: [0, 1]
                status = result[0]
                if status == 0:
                    print(f"Send  to topic `{topic}`")
                else:
                    print(f"Failed to send message to topic {topic}")
        else:
            time.sleep(0.01)

if __name__ == '__main__':
    
    #是否保存为AVI视频
    #if args.savevideo:
        #out = cv2.VideoWriter("road" + ".avi", cv2.VideoWriter_fourcc('M','J','P','G'),10,(512,256))

    print("camera connected!")        
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, 60)
    #publish(client)
    # 订阅主题
    client.subscribe(cmd_vel_topic)
    client.loop_forever()


