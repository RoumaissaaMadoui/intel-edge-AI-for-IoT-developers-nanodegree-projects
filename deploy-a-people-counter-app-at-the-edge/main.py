"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from acc import get_accuracy

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
  

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def preprocess(frame, input_shape):
    p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame

def convert_time(n):
    return time.strftime("%H:%M:%S", time.gmtime(n))

def handle_input(input_stream):
    '''
     Handle image, video or webcam
    '''
    
    # Create a flag for single images
    is_image = False
    
    # Create a flag for CAM
    is_cam = False
    
    # Check if the input is an image
    if input_stream.endswith('.jpg') or input_stream.endswith('.png') or input_stream.endswith('.bmp'):
        is_image = True
        
    # Check if the input is a webcam
    elif input_stream == 'CAM':
        is_cam = True
        
    # Check if the input is a not a video    
    elif not input_stream.endswith('.mp4'):
        log.error('Please enter a valid input!')
        
    return is_image, is_cam
    
def draw_boxes(frame, res, threshold, width, height):
    '''
    Draw Boxes
    '''
    color = (0, 255, 0)
    count = 0
    for box in res[0][0]:
        conf = box[2]
        class_id = int(box[1])
        
        # Check if confidence is bigger than the threshold and if a person is detected
        if conf >= threshold and class_id == 1:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            
            # Draw box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
            
            # Increment count
            count += 1
                
    return frame, count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    request_id = 0
    last_count = 0
    total_count = 0
    duration = 0
    start_time = 0
    end_time = 0
    detected = False
    last_six_count = []
    frame_num = 0
    
    # List to hold current_count values to calculate the accuracy
    detection_list = []
    
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    input_shape = infer_network.get_input_shape() 

    ### Handle the input stream ###
    is_image, is_cam = handle_input(args.input)
    if is_cam:
        args.input = 0
      
    ### Get and open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    ### Loop until stream is over ###
    while cap.isOpened():

        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # Increment Frame number
        frame_num += 1

        ### Pre-process the image as needed ###
        p_frame = preprocess(frame, input_shape)
        
        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame, request_id)

        ### Wait for the result ### 
        if infer_network.wait(request_id) == 0:

            ### Get the results of the inference request ###
            result = infer_network.get_output(request_id)

            ### Extract any desired stats from the results ###   
            ### Update the frame to include detected bounding boxes
            out_frame, current_count = draw_boxes(frame, result, prob_threshold, width, height)
            
            last_six_count.append(current_count)
            
            detection_list.append(current_count)
            
            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            
            ### Topic "person": keys of "count" ###
            client.publish("person", json.dumps({"count": current_count}))
                    
            if current_count > last_count and detected == False:
                start_time = frame_num 
                total_count = total_count + current_count - last_count
                detected = True
                
                ### Topic "person": keys of "total" ###
                client.publish("person", json.dumps({"total": total_count}))
                
            if current_count == 0:
   
                # Check if a person is detected in the current frame and no person was detected in the last five frames
                if (detected and all(x == 0 for x in last_six_count[-5:])):
                    detected = False 
                    
                    # Check if there was a person detected before the last five frames 
                    if(last_six_count[-6] == 1):
                    
                        # Substract the start_time and the last five frames from the current frame_num
                        end_time = frame_num - start_time - 5
                        
                        # Divide end_time by 24 to convert it to seconds, and round down to nearest integer
                        # FPS = 24
                        duration = int((end_time)/24)
                        
                        ### Topic "person/duration": key of "duration" ### 
                        client.publish("person/duration", json.dumps({"duration": duration}))
                    else:
                        pass
            
            del last_six_count[:-6] 
            last_count = current_count
            
        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()
        
        if key_pressed == 27:
                break

        ### Write an output image if `single_image_mode` ###
        if is_image:
            cv2.putText(out_frame, "current_count: {}".format(current_count), (10, height - ((1 * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite('output_image.jpg', out_frame)
    
    # I used log.warning to print accuracy to the terminal because print affect the UI behavior
    log.warning("Accuracy: {:.2f}%".format(get_accuracy(detection_list)))  
    
    # Release the capture
    cap.release()
    
    # Destroy any OpenCV windows
    cv2.destroyAllWindows
    
    # Disconnect from MQTT
    client.disconnect()



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
   
    # Connect to the MQTT server
    client = connect_mqtt()
    
    #Perform inference on the input stream
    start = time.time()
    infer_on_stream(args, client)
    end = time.time()
    dur = convert_time(end - start)
    
    # I used log.warning to print dur to the terminal because print affect the UI behavior
    log.warning("Inference time: {}".format(dur))


if __name__ == '__main__':
    main()
