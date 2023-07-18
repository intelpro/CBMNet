# CBMNet(CVPR 2023, highlight)
**Official repository for the CVPR 2023 paper, "Event-based Video Frame Interpolation with Cross-Modal Asymmetric Bidirectional Motion Fields"**

\[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Event-Based_Video_Frame_Interpolation_With_Cross-Modal_Asymmetric_Bidirectional_Motion_Fields_CVPR_2023_paper.pdf)\] 
\[[Supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Kim_Event-Based_Video_Frame_CVPR_2023_supplemental.zip)\] 
\[[Oral(YouTube)](https://www.youtube.com/watch?v=xw4pQjzoGVg)\]



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

* [[RAW-Train](http://gofile.me/5rpBN/xYrGYLjQQ)]
* [[RAW-Test](http://gofile.me/5rpBN/AuNGFI5sF)]

Also, you can download the processed data(including processed event voxel representation)

* [[Processed-Train]()]
* [[Processed-Test]()]

** Cautions: the x,y coordinates of the raw event file are multiplied by 128.

### Quick Usage

Download repository:

```bash
    $ git clone https://github.com/intelpro/CBMNet
```

Download pretrained model of our network weights

* [[Ours](https://drive.google.com/file/d/1VJKyuoRSMOJkl8fQlJIkc7S4Fmd2_X8K/view?usp=sharing)]
* [[Ours-Large](https://drive.google.com/file/d/1jI6_RwhXeM-pW5CnHf0exB5RP2zp2SbY/view?usp=sharing)]

Generate an intermediate video frame:

```bash
    $ python run.py --first images/im1.png --second images/im3.png --output images/im2.png
```

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
If you have any question, please send an email me(intelpro@kaist.ac.kr)

## License
The project codes and datasets can be used for research and education only. 