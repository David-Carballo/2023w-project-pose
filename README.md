# YOGA POSE DETECTION

<img src="images/image1.png" width="128" align="right">

A project for Postgraduate course Artificial Intelligence with Deep Learning - 2023 Winter,
    authored by **Marc Fort, Francisco Dueñas and David Carballo**. Advised by **Pol Caselles**

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#dataset">Dataset</a>
      <ul>
        <li><a href="#original-dataset">Original Dataset</a></li>
        <li><a href="#project-dataset">Project Dataset</a></li>
      </ul>
    </li>
    <li>
      <a href="#arch-models">Architecture and Models</a>
      <ul>
        <li><a href="#openpose">OpenPose</a></li>
        <li><a href="#mobilenet">MobileNet</a></li>
        <li><a href="#mlp">MLP</a></li>
        <li><a href="#raw">Raw Data</a></li>
      </ul>
    </li>
    <li><a href="#how-to">How To</a></li>
    <li><a href="#experiments">Experiments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This is a repository to introduce in Body Pose Detection, more specifically to detect Yoga Postures. The goal of this project is to learn how to manage a Deep Learning project and provide a solution that allows to improve existent solutions. using [Yoga Postures Dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset)

Goals:
* Achieve a different solution to the classification task
* Use pretrained models to generate data for later use
* Learn how to create a custom dataset that fits the needs of the project
* Create a "minimum viable product" that can be expanded upon in the future

___


<!-- DATASET -->
## Dataset

<p align="center">
    <img src="images/dataset.png" />
</p>

### Original Dataset
In this case, we have got the [Yoga Postures Dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset) that consists in 2756 images distributed in 47 classes like as shown in the following figure:

<p align="center">
    <img src="images/newplot_0.png"/>
</p>


### Transformed Dataset

To fit images with our models, we have applied different transformations and data augmentation techniques.
* Resize tensor images to [255,255] and apply normalization with mean = [0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
* Data augmentation (Probability): HorizontalFlip(50%), GaussianBlur(50%), HueSaturationValue(50%) and ColorJitter(50%)

This dataset feed the EfficientNet Model.

### Angles Dataset

To obtain the angles of the extracted poses from the original dataset, first, we have used a pretrained Open Pose model to extract all poses in a key points tensor. After that, we have applied an algorithm to compute the angles that forms each pose and its feed our Multilayer Perceptron model.


___

<!-- ARCHMODELS -->
## Architecture and Models
### OpenPose

### EfficientNet

Architecture of EfficientNet           |  EfficientNet baseline network
:-------------------------:|:-------------------------:
![](images/efficientnetarch1.png)  |  ![](images/efficientnetarch.PNG)



### MultiLayer Perceptron (MLP)

### RawData Model

### Final Classification Model


___

<!-- HOW TO -->
## How TO

All this project has been executed on Google Colab notebooks. In the next sections are provided the steps to obtain the results.

### How To Download Dataset
> Notice that it is required to create a Kaggle account in order to be able to download the dataset. 
#### &emsp;Option 1: Download from official web page
In order to download the Yoga Posture Dataset, go to its [Kaggle website page](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset), log in with your Kaggle account and then click on [download button](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/download?datasetVersionNumber=1).

![](images/yoga_web.PNG)

When you have downloaded the dataset, you must upload it in your Google Drive account to allow Colab notebooks access to it. 

#### &emsp;Option 2: Kaggle API Token
If you want load the dataset in your Google Colab, you must follow the instructions on first section of [Angles MLP Model notebook](AnglesMLP.ipynb).

In order to use the Kaggle’s public API, you must first authenticate using an API token. From the site header, click on your user profile picture, then on “My Account” from the dropdown menu. This will take you to your account settings at [Kaggle Account](https://www.kaggle.com/account). Scroll down to the section of the page labelled API:

![](images/kaggle_token.PNG)

To create a new token, click on the “Create New API Token” button. This will download a fresh authentication token onto your machine.

After that, upload the authentication token in your Google Colab files and follow the steps to extract all the dataset in your Colab. 

![](images/colabfile.PNG) 


### How To Extract Pose Keypoints
To extract image poses from datasets, we will use [Angles MLP Model notebook](AnglesMLP.ipynb). In this Colab, we find the *Compute Keypoints* section, where our code runs through all the images extracting their poses.

```bash
h 
```
![](images/pose.png)
|          ID  | Body Part  | ID | Body Part |  
|:---------------:|:---------:|:-----:|:------:|
| 0 | Head| 9 | Right Knee|
| 1 | Neck| 10 | Right Foot|
| 2 | Right Shoulder| 11 | Left Hip|
| 3 | Right Elbow| 12 | Left Knee|
| 4 | Right Hand| 13 | Left Foot|
| 5 | Left Shoulder| 14 | Right Eye|
| 6 | Left Elbow| 15 | Left Eye|
| 7 | Left Hand| 16 | Right Ear|
| 8 | Right Hip| 17 | Left Ear|

### MLP Model

### EfficientNet Model

### Final Classification Model

<!-- EXPERIMENTS -->
## Experiments
### Angles
#### Metrics
### EfficientNet vs MobileNet
#### Metrics
### Experiment3

<p align="right">(<a href="#yoga-pose-detection">back to top</a>)</p>
