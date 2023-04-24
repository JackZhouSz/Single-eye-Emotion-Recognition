# Single-eye-Emotion-Recognition
<img width="100%" src="https://github.com/zhanghaiwei1234/Single-eye-Emotion-Recognition/blob/main/img/introduce.png"></a>
Demonstration of a wearable single-eye emotion recognition prototype system consisting with a bio-inspired event-based camera (DAVIS346) and a low-power NVIDIA Jetson TX2 computing device. Event-based cameras simultaneously provide intensity and corresponding events, which we input to a newly designed lightweight Spiking Eye Emotion Network (SEEN) to effectively extract and combine spatial and temporal cues for emotion recognition. Given a sequence, SEEN takes the start and end intensity frames (green boxes) along with $n$ intermediate event frames (red boxes) as input. Our prototype system consistently recognizes emotions based on single-eye areas under different lighting conditions at $30$ FPS. 

## Documents
* More [Usages](moco-doc/usage.md)
* Detailed [HTTP APIs](moco-doc/apis.md) or [Socket APIs](moco-doc/socket-apis.md)
* Detailed [REST API](moco-doc/rest-apis.md)
* Detailed [Websocket API](moco-doc/websocket-apis.md)
* [Global Settings](moco-doc/global-settings.md) for multiple configuration files.
* [Command Line Usages](moco-doc/cmd.md)
* [Extend Moco](moco-doc/extending.md) if current API does not meet your requirement.

```bash
CUDA_VISIBLE_DEVICES=4  python main.py --root_path /video-emotion-classfication/dataset-60  --event_video_path event_30 --frame_video_path frame  --annotation_path emotion_new_adjust2.json --result_path  sometest/test_163   --dataset emotion --n_classes 7 --batch_size 32 --n_threads 16 --checkpoint 100 --inference --no_val --tensorboard --weight_decay 1e-3 --n_epochs 180 --sample_size 90 --no_hflip --sample_duration 4  --inference_batch_size 120 --inference_stride 0  --sample_t_stride 4  --inference_sample_duration 4 --thresh 0.3 --lens 0.5 --decay 0.2 --beta 0 --learning_rate 0.015 --lr_scheduler singlestep
```
