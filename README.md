# Single-eye-Emotion-Recognition
<img width="100%" src="https://github.com/zhanghaiwei1234/Single-eye-Emotion-Recognition/blob/main/img/introduce.png"></a>
```bash
CUDA_VISIBLE_DEVICES=4  python main.py --root_path /video-emotion-classfication/dataset-60  --event_video_path event_30 --frame_video_path frame  --annotation_path emotion_new_adjust2.json --result_path  sometest/test_163   --dataset emotion --n_classes 7 --batch_size 32 --n_threads 16 --checkpoint 100 --inference --no_val --tensorboard --weight_decay 1e-3 --n_epochs 180 --sample_size 90 --no_hflip --sample_duration 4  --inference_batch_size 120 --inference_stride 0  --sample_t_stride 4  --inference_sample_duration 4 --thresh 0.3 --lens 0.5 --decay 0.2 --beta 0 --learning_rate 0.015 --lr_scheduler singlestep
```
