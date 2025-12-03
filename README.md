<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Vegas-City-Road (2025/12/03)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Aerial Imagery Vegas-City Road</b> (Singleclass)  based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512  pixels 
<a href="https://drive.google.com/file/d/1ErOkrNOnOOvyIP16YuOunXsICZ9Uis1u/view?usp=sharing">
<b>Tiled-Vegas-City-Road-ImageMask-Dataset.zip </b></a>
which was derived by us from 
<a href="https://www.kaggle.com/datasets/yash0309/aerial-images-of-vegas-city">
<b>road_network_Detection_Vegas_city_Dataset</b> 
</a> on the kaggle web site 
<br><br>
<b>Divide-and-Conquer Strategy</b><br>
The kaggle web site <b>Vegas-City-Road</b> dataset features a mask pixel size of 1300 x 1300 pixels, which is different from the corresponding image pixel
 size of 512 x 512. Therefore, we first generated a 1024 x 1024 pixel <b>Resized image and mask master</b> dataset from the original Vegas dataset,
 and then 
adopted the following <b>Divide-and-Conquer Strategy</b> to build our Vegas-City-Road segmentation model.
<br>
<br>
Please see also our experiment <a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Massachusetts-Road">
TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Massachusetts-Road</a>
<br>
<br>
<b>1. Tiled Image Mask Dataset</b><br>
We generated a 512 x 512 pixels tiledly-split dataset from the 1024 x 1024 pixels Resized master by our offline augmentation tool <a href="./generator/TiledImageMaskDatasetGenerator.py">
TiledImageMaskDatasetGenerator</a>
<br>
<br>
<b>2. Train Model by Tiled ImageMask Dataset</b><br>
We trained and validated the TensorFlowFlexUNet model for Aerial Imagery Vegas-City Roads by using the 
Tiled-Vegas-City Road dataset.
<br><br>
<b>3. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict mask regions for the mini_test images 
with the resolution of 1024 x 1024 pixels.
<br><br>
<hr>
<b>Actual Image Segmentation for Vegas-City Road Images of 1024 x 1024 pixels</b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/images/10217.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/masks/10217.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test_output_tiled/10217.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/images/10220.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/masks/10220.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test_output_tiled/10220.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/images/10247.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/masks/10247.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test_output_tiled/10247.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from <a href="https://www.kaggle.com/datasets/yash0309/aerial-images-of-vegas-city">
<b>road_network_Detection_Vegas_city_Dataset</b> 
</a> on the kaggle web site 
<br><br>
<b>About Dataset</b><br>
This dataset contains high-resolution aerial images of Vegas City, along with corresponding annotated binary masks designed for road network detection tasks. The dataset is organized into training, validation, and testing sets, making it suitable for developing and evaluating deep learning models for semantic segmentation. <br>
It has been effectively used with the U-Net architecture, which excels in tasks requiring precise pixel-level classification.<br>
 This dataset can support applications in urban planning, autonomous driving, and geospatial analysis.
<br><br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by-sa/4.0/">
CC BY-SA 4.0
</a>
<br>
<br>
<h3>
2 Tiled Vegas-City Road ImageMask Dataset
</h3>
 If you would like to train this Vegas-City Road Segmentation model by yourself,
 please download <a href="https://drive.google.com/file/d/1ErOkrNOnOOvyIP16YuOunXsICZ9Uis1u/view?usp=sharing">
 <b>Tiled-Vegas-City-Road-ImageMask-Dataset.zip (4.4GB)</b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Vegas-City-Road
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Tiled-Vegas-City-Road Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Vegas-City-Road/Vegas-City-Road_Statistics.png" width="512" height="auto"><br>
<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br> 
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNetModel
</h3>
 We trained Vegas-City-Road TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Vegas-City-Road/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Vegas-City-Road and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Vegas-City-Road 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;                      Road: white
rgb_map = {(0,0,0):0, (255,255,255):1,}
</pre>

<b>Tiled inference parameters</b><br>
<pre>
[tiledinfer] 
overlapping = 128
images_dir    = "./mini_test/images/"
output_dir    = "./mini_test_output_tiled/"
</pre>

<b>Epoch change tiled inference callback</b><br>
Enabled <a href="./src/EpochChangeTiledInferencer.py">epoch_change_tiled_infer callback (EpochChangeTiledInferencer.py)</a></b>.<br>
<pre>
[train]
poch_change_infer     = False
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = True
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>

By using this callback, on every epoch_change, the tiled inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 13,14,15)</b><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 26,27,28)</b><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 28 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/train_console_output_at_epoch28.png" width="880" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Vegas-City-Road/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Vegas-City-Road/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Vegas-City-Road</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Vegas-City-Road.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/evaluate_console_output_at_epoch28.png" width="880" height="auto">
<br><br>Image-Segmentation-Aerial-Imagery

<a href="./projects/TensorFlowFlexUNet/Vegas-City-Road/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Vegas-City-Road/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.1269
dice_coef_multiclass,0.9326
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Vegas-City-Road</b> folder, and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Vegas-City-Road.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled_inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Vegas-City-Road Images of 1024 x 1024 pixels </b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/images/10221.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/masks/10221.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test_output_tiled/10221.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/images/10223.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/masks/10223.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test_output_tiled/10223.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/images/10225.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/masks/10225.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test_output_tiled/10225.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/images/10238.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/masks/10238.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test_output_tiled/10238.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/images/10244.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/masks/10244.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test_output_tiled/10244.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/images/10252.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test/masks/10252.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Vegas-City-Road/mini_test_output_tiled/10252.png" width="320" height="auto"></td>
</tr>


</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Satellite Imagery Aerial-Imagery Segmentation</b><br>
Nithish<br>
<a href="https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812">
https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812
</a>
<br>
<br>
<b>2. Deep Learning-based Aerial-Imagery Segmentation Using Aerial Images: A Comparative Study</b><br>
Kamal KC, Alaka Acharya, Kushal Devkota, Kalyan Singh Karki, and Surendra Shrestha<br>
<a href="https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study">
https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study</a>
<br>
<br>
<b>3. A Comparative Study of Deep Learning Methods for Automated Aerial-Imagery Network<br>
Extraction from High-Spatial-ResolutionRemotely Sensed Imagery</b><br>
Haochen Zhou, Hongjie He, Linlin Xu, Lingfei Ma, Dedong Zhang, Nan Chen, Michael A. Chapman, and Jonathan Li<br>
<a href="https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf">
https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf
</a>
<br>
<br>
<b>4. Machine Learning for Aerial Image Labeling</b><br>
Volodymyr Mnih<br>
<a href="https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf">
https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf
</a>
<br>
<br>
<b>5.City-Scale Road Extraction from Satellite Imagery v2:
Road Speeds and Travel Times</b><br>
Adam Van Etten<br>
<a href="https://openaccess.thecvf.com/content_WACV_2020/papers/Van_Etten_City-Scale_Road_Extraction_from_Satellite_Imagery_v2_Road_Speeds_and_WACV_2020_paper.pdf">
https://openaccess.thecvf.com/content_WACV_2020/papers/Van_Etten_City-Scale_Road_Extraction_from_Satellite_Imagery_v2_Road_Speeds_and_WACV_2020_paper.pdf
</a>
<br>
<br>

<b>6. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>7. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Massachusetts-Road</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Massachusetts-Road">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery-Massachusetts-Road
</a>
<br>
<br>

