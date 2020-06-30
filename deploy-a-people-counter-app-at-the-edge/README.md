# Deploy a People Counter App at the Edge

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.5 or 3.6 |

![people-counter-python](./images/people-counter-image.png)

## What it Does

The people counter application will demonstrate how to create a smart video IoT solution using Intel® hardware and software tools. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count.

## How it Works

The counter will use the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. The model used should be able to identify people in a video frame. The app should count the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. It then sends the data to a local web server using the Paho MQTT Python package.

You will choose a model to use and convert it with the Model Optimizer.

![architectural diagram](./images/arch_diagram.png)

## Requirements

### Hardware

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2)
* OR Udacity classroom workspace for the related course

### Software

*   Intel® Distribution of OpenVINO™ toolkit 2019 R3 release
*   Node v6.17.1
*   Npm v3.10.10
*   CMake
*   MQTT Mosca server
  
        
## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Utilize the classroom workspace, or refer to the relevant instructions for your operating system for this step.

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)

### Install Nodejs and its dependencies

Utilize the classroom workspace, or refer to the relevant instructions for your operating system for this step.

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)

### Install npm

There are three components that need to be running in separate terminals for this application to work:

-   MQTT Mosca server 
-   Node.js* Web server
-   FFmpeg server
     
From the main directory:

* For MQTT/Mosca server:
   ```
   cd webservice/server
   npm install
   ```

* For Web server:
  ```
  cd ../ui
  npm install
  ```
  **Note:** If any configuration errors occur in mosca server or Web server while using **npm install**, use the below commands:
   ```
   sudo npm install npm -g 
   rm -rf node_modules
   npm cache clean
   npm config set registry "http://registry.npmjs.org"
   npm install
   ```
## Model Research
In investigating potential people counter models, I tried different models. At the end, I have decided to use `person-detection-retail-0013` because it was fast and accurate.

### Pre-trained models (OpenVINO Zoo)
Download `person-detection-retail-0013`

```
sudo /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 --precisions FP32
```

### TensorFlow Models
Download the model
```
wget model_link
```
Unzipp the model
```
tar -xvf model_name.tar.gz
```
cd to the model folder
```
cd model_name
```
Convert models to IR
1. [faster_rcnn_resnet50](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)
2. [faster_rcnn_resnet101](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)
3. [faster_rcnn_inception_v2](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)

   ```
   python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py  \
   --input_model frozen_inference_graph.pb \
   --reverse_input_channels \
   --data_type FP16 \
   --tensorflow_object_detection_api_pipeline_config pipeline.config \
   --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
   ```
4. [rfcn_resnet101](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz)
  
   ```
   python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py  \
   --input_model frozen_inference_graph.pb \
   --reverse_input_channels \
   --data_type FP16 \
   --tensorflow_object_detection_api_pipeline_config pipeline.config \
   --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/rfcn_support.json
   ``` 
5. [ssdlite_mobilenet_v2](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
  
   ```
   python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py  \
   --input_model frozen_inference_graph.pb \
   --reverse_input_channels \
   --tensorflow_object_detection_api_pipeline_config pipeline.config \
   --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
   ```
Reasons for not choosing the above models  
- Model 1 to 4:
  - The models were insufficient for the app because **they were slow and took too much time to complete inference on the video.**
  - I tried to improve the models for the app by **using one of the optimization techniques "quantization" and that by reducing precision to FP16**.
- Model 5:
  - The model was insufficient for the app because **it wasn't accurate, the model didn't detect the second person in the video for more than 157 frames.**
  - I tried to improve the model for the app by **trying lower confidence threshold values.**
  
## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were:
- **Accuracy**: (TP+TN))/total, I have used confusion matrix.
- **Size of the model**: compare the file size of .pb (Tensorflow) and .bin (IR).
- **Inference time**: time to complete inference on the video.

> Pre-conversion calculations were done in `pre_conversion_accuracy_inference_time.ipynb` notebook.

<table style="width:100%">
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Pre-conversion size</th>
    <th colspan="3">Post-conversion size</th> 
  </tr>
  <tr>
  	<th>Size (MB)</th>
  	<th>Accuracy (%)</th>
  	<th>Inference time (hh:mm:ss)</th>
  	<th>Size (MB)</th>
  	<th>Accuracy (%)</th>
  	<th>Inference time (hh:mm:ss)</th>
  </tr>
  <tr>
  	<td>rfcn_resnet101</td>
    <td>208</td>
    <td>98.28</td>
    <td>01:48:50</td>
    <td>102</td>
    <td>95.77</td>
    <td>01:01:53</td>
  </tr>
  <tr>
  	<td>ssdlite_mobilenet_v2</td>
    <td>19</td>
    <td>84.43</td>
    <td>00:02:26</td>
    <td>17.1</td>
    <td>79.05</td>
    <td>00:02:15</td>
  </tr>
  <tr>
  	<td>faster_rcnn_resnet50</td>
    <td>115</td>
    <td>90.10</td>
    <td>01:31:23</td>
    <td>55.6</td>
    <td>95.41</td>
    <td>01:15:36</td>
  </tr>
   <tr>
  	<td>faster_rcnn_resnet101</td>
    <td>187</td>
    <td>97.78</td>
    <td>02:05:23</td>
    <td>91.8</td>
    <td>98.57</td>
    <td>01:32:47</td>
  </tr>
   <tr>
  	<td>faster_rcnn_inception_v2</td>
    <td>54.5</td>
    <td>97.27</td>
    <td>00:32:29</td>
    <td>25.4</td>
    <td>97.27</td>
    <td>00:23:05</td>
  </tr>
   <tr>
    <td><b>person-detection-retail-0013 (Best Model)</b></td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>2.75</td>
    <td>97.99</td>
    <td>00:02:42</td>
  </tr>
</table>
Compare the differences in network needs and costs of using cloud services as opposed to deploying at the edge: 

- **Cloud services** are expensive and needs high network bandwidth.
- **Deploying at the edge** is not expensive and has less network bandwidth requirement, because it processes, analyses, and performs necessary actions on the collected data locally, there is no need to transfer gigabytes of data to the cloud.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
- In labs or any place where people should not get exposed to radiation for a long time. If any person spends more than the allowed time they get a warning. The app will be useful because it calculates the time each person spends in the room.
- In Hospital rooms to calculate the number of people and time spend in room while visiting a patient. If total_count and duration are more than the allowed they get a warning. The app will be useful because it calculates total_count and duration.
- In Shopping Mall to count the number of people visited a specific shop. The app will be useful because it calculates the total_count.

## Run the application

From the main directory:

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
Mosca server started.
```

### Step 2 - Start the GUI

Open new terminal and run below commands.
```
cd webservice/ui
npm run dev
```

You should see the following message in the terminal.
```
webpack: Compiled successfully
```

### Step 3 - FFmpeg Server

Open new terminal and run the below commands.
```
sudo ffserver -f ./ffmpeg/server.conf
```

### Step 4 - Run the code

Open a new terminal to run the code. 

#### Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

You should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.

#### Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at: 

```
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/
```

*Depending on whether you are using Linux or Mac, the filename will be either `libcpu_extension_sse4.so` or `libcpu_extension.dylib`, respectively.* (The Linux filename may be different if you are using a AVX architecture)

Though by default application runs on CPU, this can also be explicitly specified by ```-d CPU``` command-line argument:

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```
If you are in the classroom workspace, use the “Open App” button to view the output. If working locally, to see the output on a web based interface, open the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) in a browser.

#### Running on the Intel® Neural Compute Stick

To run on the Intel® Neural Compute Stick, use the ```-d MYRIAD``` command-line argument:

```
python3.5 main.py -d MYRIAD -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

To see the output on a web based interface, open the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) in a browser.

**Note:** The Intel® Neural Compute Stick can only run FP16 models at this time. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

#### Using a camera stream instead of a video file

To get the input video from the camera, use the `-i CAM` command-line argument. Specify the resolution of the camera using the `-video_size` command line argument.

For example:
```
python main.py -i CAM -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

To see the output on a web based interface, open the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) in a browser.

**Note:**
User has to give `-video_size` command line argument according to the input as it is used to specify the resolution of the video or image file.

## A Note on Running Locally

The servers herein are configured to utilize the Udacity classroom workspace. As such,
to run on your local machine, you will need to change the below file:

```
webservice/ui/src/constants/constants.js
```

The `CAMERA_FEED_SERVER` and `MQTT_SERVER` both use the workspace configuration. 
You can change each of these as follows:

```
CAMERA_FEED_SERVER: "http://localhost:3004"
...
MQTT_SERVER: "ws://localhost:3002"
```
