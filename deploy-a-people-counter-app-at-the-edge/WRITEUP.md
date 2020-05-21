# Project Write-Up

## Explaining Custom Layers
Custom layers are layers that are not included into a [list of known layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html).

The process behind converting custom layers involves adding extensions to both:
1. **The Model Optimizer** : needs two necessary custom layer extensions
   - Custom Layer Extractor 
   - Custom Layer Operation
2. **The Inference Engine**: needs two custom layer extensions
   - Custom Layer CPU extension
   - Custom Layer GPU Extension

We use model extension generator to generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine. To complete the implementation of each extension, the template functions may need to be edited to fill-in details specific to the custom layer or the actual custom layer functionality itself.

Some of the potential reasons for handling custom layers are **to avoid errors reported by the Inference Engine if a model topology contains customs layers**

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were:
- **Accuracy**: (TP+TN))/total, I have used confusion matrix.
- **Size of the model**: compare the file size of .pb (Tensorflow) and .bin (IR).
- **Inference time**: time to complete inference on the video.

> Pre-conversion calculations are done in `pre_conversion_accuracy_inference_time.ipynb` notebook.

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

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
- **Lighting:** if there isn't a sufficient amount of light the model can't predict in the dark.
- **Model accuracy:** Since the decisions need to be made so fast the model need to be accurate there is no place to make mistakes.
- **Camera focal**: The longer the focal length, the narrower the angle of view and the higher the magnification. The shorter the focal length, the wider the angle of view and the lower the magnification. So camera focal need to be chosen based on the user needs.
- **Length/Image size**: if the size was small the model can't predict accurately and if the size was large the model will take a long time to finish.


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
