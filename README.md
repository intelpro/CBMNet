# CBMNet(CVPR 2023, highlight)
**Official repository for the CVPR 2023 paper, "Event-based Video Frame Interpolation with Cross-Modal Asymmetric Bidirectional Motion Fields"**

\[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Event-Based_Video_Frame_Interpolation_With_Cross-Modal_Asymmetric_Bidirectional_Motion_Fields_CVPR_2023_paper.pdf)\] 
\[[Supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Kim_Event-Based_Video_Frame_CVPR_2023_supplemental.zip)\] 



## Qualitative video demos on ERF-X170FPS dataset 
### Falling pop-corn
<img src="https://github.com/intelpro/CBMNet/raw/main/figure/popcorn.gif" width="100%" height="100%">
<!--
![real_event_045_resized](/figure/video_results_real_event3.gif "real_event_045_resized")
-->

### Flowers
<img src="https://github.com/intelpro/CBMNet/raw/main/figure/flower.gif" width="100%" height="100%">
<!--
![real_event_045_resized](/figure/video_results_real_event3.gif "real_event_045_resized")
-->

### Driving scene
<img src="https://github.com/intelpro/CBMNet/raw/main/figure/driving.gif" width="100%" height="100%">
<!--
![real_event_045_resized](/figure/video_results_real_event3.gif "real_event_045_resized")
-->

## ERF-X170FPS dataset
#### Dataset of high-resolution (1440x975), high-fps (170fps) video frames plus high resolution events with extremely large motion using the beam-splitter acquisition system:
![image info](./figure/ERF_dataset.png)

### Quantitative results on the ERF-X170FPS datasets
<img src="https://github.com/intelpro/CBMNet/raw/main/figure/Quantitative_eval_ERF_x170FPS.png" width="60%" height="60%">


### Downloading ERF-X170FPS datasets 
You can download the raw-data(collected frame and events) from this links

* [[RAW-Train](https://drive.google.com/file/d/1Bsf9qreziPcVEuf0_v3kjdPUh27zsFXK/view?usp=drive_link)]
* [[RAW-Test](https://drive.google.com/file/d/1Dk7jVQD29HqRVV11e8vxg5bDOh6KxrzL/view?usp=drive_link)]

** Cautions: the x,y coordinates of the raw event file are multiplied by 128.

## Requirements
* PyTorch 1.8.0
* CUDA 11.2
* python 3.8

## Quick Usage

Download repository:

```bash
    $ git clone https://github.com/intelpro/CBMNet
```

Install correlation package:

```bash
    $ sh install_correlation.sh
```

Download network weights(trained on ERF-X170FPS datasets) and place downloaded model in ./pretrained_model/

* [[Ours](https://drive.google.com/file/d/1VJKyuoRSMOJkl8fQlJIkc7S4Fmd2_X8K/view?usp=sharing)]
* [[Ours-Large](https://drive.google.com/file/d/1jI6_RwhXeM-pW5CnHf0exB5RP2zp2SbY/view?usp=sharing)]


Generate an intermediate video frame using ours model:

```bash
    $ python run_samples.py  --model_name ours --ckpt_path pretrained_model/ours_weight.pth --save_output_dir ./output --image_number 0
```

Also, you can generate intermediate video frame using ours-large model:

```bash
    $ python run_samples.py  --model_name ours_large --ckpt_path pretrained_model/ours_large_weight.pth --save_output_dir ./output --image_number 0

```


## 🚀 Quick Test on BSERGB Dataset

This section describes how to test the model on the **BSERGB dataset** using the pre-trained weights.

### 1. Download BSERGB Dataset

You can download the [BSERGB dataset](https://github.com/uzh-rpg/timelens-pp) from the official TimeLens++ GitHub repository.

### 2. Preprocess Event Voxel Data

After downloading, the BSERGB dataset should have the following directory structure:


```
├── BSERGB/
│   ├── 1_TEST/
│   │   ├── scene_001/
│   │   │   ├── images/
│   │   │   │   ├── 000000.png
│   │   │   │   ├── ...
│   │   │   ├── events/
│   │   │   │   ├── 000000.npz
│   │   │   │   ├── ...
│   │   ├── scene_002/
│   │   ├── scene_003/
│   │   ├── ...
│   ├── 2_VALIDATION/
│   │   ├── scene_001/
│   │   ├── scene_002/
│   │   ├── ...
│   ├── 3_TRAINING/
│   │   ├── scene_001/
│   │   ├── scene_002/
│   │   ├── ...
```

Now, convert the raw event data into event voxel grids using the following command:


```bash
    $ python tools/preprocess_events.py  --dataset_dir BSERGB_DATASET_DIR --mode 1_TEST

```
- ``--dataset_dir BSERGB_DATASET_DIR``: Specifies the BSERGB dataset directory.
- ``--mode 1_TEST``: Select the mode to convert raw events into event voxels. Choose 1_TEST if you only want to perform testing.

### 🛠️ Event Voxel Preprocessing Output

After preprocessing, event voxel files will be generated and saved into the **target folder**.  
For each sample, three types of voxel grids will be saved:

- `0t`: Events from the start frame to the interpolated frame  
- `t0`: Reversed version of `0t` (used for backward flow)  
- `t1`: Events from the interpolated frame to the end frame  

Each event voxel is stored in the following naming format:

<start_frame_idx>-<interp_frame_idx>-<end_frame_idx>_{suffix}.npz

Each index is zero-padded using `zfill(6)`. The `{suffix}` represents one of the three types: `0t`, `t0`, or `t1`.

#### 📁 Example

```text
000000-000002-000004_0t.npz   # event voxel from 000000 to 000002  
000000-000002-000004_t0.npz   # reversed event voxel from 000002 to 000000  
000000-000002-000004_t1.npz   # event voxel from 000002 to 000004
```

Once the voxel preprocessing is complete and the files are generated in the proper format, you can proceed to download the pretrained model and run the test script.

### 3. Download Pretrained Weights

Download the our-large weights (trained on the BSERGB dataset) and place the downloaded model inside the ./pretrained_model directory.

🔗 **[Ours-Large(BSERGB)](https://drive.google.com/file/d/1T5ycqQK4KVZQ4pAnNkr2Ff8XIWvhmj_f/view?usp=sharing)**

Then move the file to the `./pretrained_model` directory:

``` bash
# Ensure the directory exists
mkdir -p pretrained_model

# Move the downloaded model to the correct location
mv /path/to/downloaded/Ours_Large_BSERGB.pth ./pretrained_model/
```

Make sure the final path is:


``` bash
./pretrained_model/Ours_Large_BSERGB.pth
```

### 4. Run test scripts

Once preprocessing and downloading the pretrained model are complete, you can test the model on the BSERGB dataset:

``` bash
    $ python test_bsergb.py  --dataset_dir BSERGB_DATASET_DIR
```

After running this script, gt and result images will be generated inside the ./output directory.

By evaluating the output images, you can reproduce the same quantitative results reported in our paper.

## 🚀 Train model on BSERGB Dataset

Training instructions and documentation will be available in the near future. (Work in progress)

In the meantime, if you need to proceed quickly, please refer to the `train.py` file for rough guide.


## Reference
> Taewoo Kim, Yujeong Chae, Hyun-kyurl Jang, and Kuk-Jin Yoon" Event-based Video Frame Interpolation with Cross-modal Asymmetric Bidirectional Motion Fields", In _CVPR_, 2023.
```bibtex
@InProceedings{Kim_2023_CVPR,
    author    = {Kim, Taewoo and Chae, Yujeong and Jang, Hyun-Kurl and Yoon, Kuk-Jin},
    title     = {Event-Based Video Frame Interpolation With Cross-Modal Asymmetric Bidirectional Motion Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {18032-18042}
}
```
## Contact
If you have any question, please send an email to taewoo(intelpro@kaist.ac.kr)

## License
The project codes and datasets can be used for research and education only. 
