# Surface Normal Estimation of Tilted Images via Spatial Rectifier

# Abstract

We present a spatial rectifier to estimate surface normal of tilted images. 
Tilted images are of particular interest as more visual data are captured by embodied sensors such as body-/robot-mounted cameras. Existing approaches exhibit bounded performance on predicting surface normal because they were trained by gravity-aligned images. 
Our two main hypotheses are: (1) visual scene layout is indicative of the gravity direction; and 
(2) not all surfaces are equally represented, i.e., there exists a transformation of the tilted image to produce better surface normal prediction. 
We design the spatial rectifier that is learned to transform the surface normal distribution of the tilted image such that it matches to that of gravity aligned images. 
The spatial rectifier is parametrized by a principle direction of the rectified image that maximizes the distribution match. 
When training, we jointly learn the gravity and principle direction that can synthesize images and surface normal to supervise the surface normal estimator. 
Inspired by the panoptic feature pyramid network, a new network that can access to both global and local visual features is proposed by leveraging dilated convolution in the decoder. The resulting estimator produces accurate surface normal estimation, outperforming state-of-the-art methods and data augmentation baseline. 
We evaluate our method not only on ScanNet and NYUv2 but also on a new dataset called Tilt-RGBD that includes substantial roll and pitch camera motion captured by body-mounted cameras.

# Installation Guide
For convenience, all the code in this repositority are assumed to be run inside NVIDIA-Docker. 

### For instructions on installing NVIDIA-Docker, please follow the following steps (note that this is for Ubuntu 18.04):

For more detailed instructions, please refer to [this link](https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/).
1. Install Docker

    ```
    sudo apt-get update
    
    sudo apt-get install \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common
        
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    
    sudo apt-key fingerprint 0EBFCD88
    
    sudo add-apt-repository \
       "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"
    
    sudo apt-get update
    
    sudo apt-get install docker-ce docker-ce-cli containerd.io
    ```
    
    To verify Docker installation, run:

    ```
    sudo docker run hello-world
    ```

2. Install NVIDIA-Docker

    ```
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list
      
    sudo apt-get update
    
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -
    
    sudo apt-get install nvidia-docker2
    
    sudo pkill -SIGHUP dockerd
    ```

To activate the docker environment, run the following command:

```
nvidia-docker run -it --rm --ipc=host -v /:/mars nvcr.io/nvidia/pytorch:19.08-py3
```

where `/` is the directory in the local machine (in this case, the root folder), and `/mars` is the reflection of that directory in the docker. 
This has also specified NVIDIA-Docker with PyTorch version 19.08 which is required to ensure the compatibility 
between the packages used in the code (at the time of submission).

Inside the docker, change the working directory to this repository: 
```
cd /mars/PATH/TO/THIS/REPO/SurfaceNormalEstimation_release
```

# Datasets and pretrained models

### Datasets

For each dataset below, please follow the provided link to download and extract them to [`./datasets/`](./datasets).

1. **ScanNet**: for more information on downloading ScanNet dataset which contains ground-truth surface normal computed by FrameNet, 
please refer to [FrameNet's repo](https://github.com/hjwdzh/FrameNet/tree/master/src).

2. **NYUv2**: follow this [link](https://www.dropbox.com/s/eape2vv26fdr0yt/nyu-normal.zip?dl=0) to download, extract, and place the `nyu-normal` folder inside [`./datasets/`](./datasets).

3. **Tilt-RGBD** (also called **KinectAzure** in the code): follow this [link](https://www.dropbox.com/s/2c9xcy1ru93m44c/KinectAzure.zip?dl=0) to download, extract, and place the `KinectAzure` folder inside [`./datasets/`](./datasets).

### Datasets splits

In order to obtain the exact dataset splits that we used for training/testing, 
please follow this [link](https://www.dropbox.com/s/qzi25m4iuyb2pxi/data.zip?dl=0) to download, extract, and place the `.pkl` files inside [`./data/`](./data).

### Pretrained models

We provide the checkpoints for all the experimental results reported in the paper with different combinations of 
network architecture, loss function, method, and training dataset. 
Please follow this [link](https://www.dropbox.com/s/ir3cxsgd30c027v/checkpoints.zip?dl=0) to download, extract, and place the `.ckpt` files inside [`./checkpoints/`](./checkpoints).

# Quick Inference

Please follow the below steps to extract surface normals from some RGB images using our provided pre-trained model:
1. Make sure you have the following `.ckpt` files inside [`./checkpoints/`](./checkpoints) folder: 
`DFPN_TAL_SR.ckpt`, `FPN_canonical_view.ckpt`, `FPN_generalized_view.ckpt`, `FPN_warping_params.ckpt`.
You can also follow this [link](https://www.dropbox.com/s/waje7724dwesmkr/demo_checkpoints.zip?dl=0) to download ONLY these checkpoints.

2. Download our demo RGB images from this [link](https://www.dropbox.com/s/y09y86x2ywtwafx/demo_dataset.zip?dl=0), extract, and place the `.png` files inside [`./demo_dataset/`](./demo_dataset).

3. Run [`inference.sh`](./inference.sh) to extract the results in [`./demo_results/`](./demo_results).


# Benchmark Evaluation
We evaluate surface normal estimation on ScanNet, NYUD-v2, or Tilt-RGBD with different network architectures using our provided pre-trained models.

Run:
```
sh evaluate.sh
```


Specifically, inside the bash script, multiple arguments are needed, including the path to the pre-trained model, folder containing log files, batch size, network architecture, and test dataset (ScanNet/NYUv2/Tilt-RGBD).
Please refer to the actual code for the exact supported arguments options.

**(Note: make sure you specify the correct network architecture for your pretrained model)**
```
CUDA_VISIBLE_DEVICES=0 python evaluate_surface_normal.py --checkpoint_path 'PATH_TO_PRETRAINED_MODEL' \
                                                         --log_folder 'PATH_TO_FOLDER_SAVING_RESULTS' \
                                                         --batch_size BATCH_SIZE \
                                                         --net_architecture 'NETWORK_ARCHITECTURE' \
                                                         --test_dataset 'TEST_DATASET'
```

# Training

We train our surface normal estimation network on ScanNet dataset. 
We will update the code for training the full *Spatial Rectifier* network pipeline in the future.

Run:
```
sh train.sh
```

Specifically, inside the bash script, multiple arguments are needed, including the folder containing log files and checkpoints,
type of loss functions (L2, AL, or TAL), learning rate, batch size, network architecture (DORN/DFPN/P-FPN), training/testing/validation datasets (ScanNet/NYUv2/Tilt-RGBD). 
Please refer to the actual code for the exact supported arguments options.

**(Note: make sure you specify the correct network architecture for your pretrained model)**


```
CUDA_VISIBLE_DEVICES=0 python train_surface_normal.py --log_folder 'PATH_TO_FOLDER_SAVING_RESULTS' \
                                                      --operation 'LOSS_FUNCTION' \
                                                      --learning_rate LEARNING_RATE \
                                                      --batch_size BATCH_SIZE \
                                                      --net_architecture 'NETWORK_ARCHITECTURE' \
                                                      --train_dataset 'TRAIN_DATASET' \
                                                      --test_dataset 'TEST_DATASET' \
                                                      --test_dataset 'TEST_DATASET' \
                                                      --val_dataset 'VALIDATION_DATASET'
```
# 






