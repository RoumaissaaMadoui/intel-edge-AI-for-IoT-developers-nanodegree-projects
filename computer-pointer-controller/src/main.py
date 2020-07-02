
import sys
import cv2
import logging as log

from argparse import ArgumentParser
from input_feeder import InputFeeder

from face_detection import Model_face_detection



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--fd_model", required=True, type=str,
                        help="Path to an xml of the Face Detection model.")
   
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    
    parser.add_argument("-dis", "--display", required=False, default=True, type=str,
                        help="Flag to display the outputs of the intermediate models")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
    return parser

def handle_input_type(input_stream):
    '''
     Handle image, video or webcam
    '''
    
    # Check if the input is an image
    if input_stream.endswith('.jpg') or input_stream.endswith('.png') or input_stream.endswith('.bmp'):
        input_type = 'image'
        
    # Check if the input is a webcam
    elif input_stream == 'CAM':
        input_type = 'cam'
        
    # Check if the input is a video    
    elif input_stream.endswith('.mp4'):
        input_type = 'video'
    else: 
        log.error('Please enter a valid input! .jpg, .png, .bmp, .mp4, CAM')
        sys.exit()    
    return input_type

def infer_on_stream(args):
    """
    Initialize the inference networks, stream video to network,
    and output stats, video and control the mouse pointer.

    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    
    # Initialise the classes
    face_detection_network = Model_face_detection(args.fd_model, args.device)

    # Load the models 
    face_detection_network.load_model()

    # Handle the input stream
    input_type = handle_input_type(args.input)
    
    # Initialise the InputFeeder class
    feed = InputFeeder(input_type=input_type, input_file=args.input)
    
    # Get the video capture
    cap = feed.load_data()

    #Loop until stream is over 
    while cap.isOpened():

        # Read from the video capture 
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # Run inference on the models     
        out_frame, face, face_coords = face_detection_network.predict(frame, args.prob_threshold, args.display)
        
        if key_pressed == 27:
            break
       
       # Display the resulting frame
        cv2.imshow('Visualization', cv2.resize(out_frame,(600,400)))

    # Release the capture
    cap.release()
    
    # Destroy any OpenCV windows
    cv2.destroyAllWindows
 

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    #Perform inference on the input stream
    infer_on_stream(args)

if __name__ == '__main__':
    main()
