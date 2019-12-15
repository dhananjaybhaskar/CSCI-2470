## CSCI-2470
### Class Project: A de-weathering extension for self-driving cars

We will implement a novel architecture inspired from existing state-of-the-art methods for automated removal of artifacts due to weather in images. We envision that our algorithm will be used to preprocess images in real-time in order to increase the robustness of object detection & classification for self-driving cars. One of the challenges associated with this project is the acquisition of training dataset. We propose generating synthetic training images by performing realistic simulation and rendering of rain and snow. To test our algorithm, we will use dashcam videos available online.

**How to train:**

Download the real_world_rain_dataset_CVPR19 from https://stevewongv.github.io/derain-project.html

python train.py --data 'path/to/real_world.txt'

**Relevant Papers:**

- Wang et al. [Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset](https://arxiv.org/abs/1904.01538)
- Liu et al. [DesnowNet: Context-Aware Deep Network for Snow Removal](https://arxiv.org/pdf/1708.04512.pdf)
- Lehtinen et al. [Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189)

**Appendix**

We used [youtube-dl](https://ytdl-org.github.io/youtube-dl/index.html), FFmpeg and ImageMagick to create animated gifs and movies:

To crop 1 min starting at 25 mins, from webm video downloaded using youtube-dl:

`ffmpeg -ss 00:25:00 -i mov.webm -ss 00:01:00 -t 00:01:00 -c copy mov1_clipped.webm`

Convert to more portable mp4 format:

`ffmpeg -i mov1_clipped.webm -max_muxing_queue_size 9999 mov1_clipped.mp4`

Extract image frames from mp4 file:

`mkdir imgs`

`ffmpeg -i mov1_clipped.mp4 imgs/%04d.png`

Resize all images and crop out 256x256 patch:

`for f in *.png; do convert $f -resize 20% $f; done`

`for f in *.png; do convert $f -crop 256x256+256+88 $f; done`

Create video from image files starting with image number 130:

`ffmpeg -r 60 -f image2 -s 256x256 -start_number 130 -i %04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p my_movie.mp4`
