# groot-isaaclab-simulator

这个项目用于在 IsaacLab 环境中调用 GR00T 推理模型，让机器人根据自然语言指令执行任务。

## 快速启动

建议分别打开两个终端：一个终端启动 GR00T 推理服务，另一个终端启动 IsaacLab 仿真并连接推理服务。

### 1. 启动推理模型

在项目根目录下使用.venv运行：

```bash
python ./gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/GR00T-N1.7-3B \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --port 5555
```
该命令会加载 `checkpoints/GR00T-N1.7-3B` 模型，并在 `5555` 端口启动推理服务。

### 2. 启动 IsaacLab 并开始推理，都要在conda的env_isaaclab环境运行

推理服务启动后，在另一个终端运行测试推理环境：
```bash
~/Project/IsaacLab/isaaclab.sh -p ~/Project/Isaac-GR00T/examples/IsaacLab/isaaclab_gr00t_adapter.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --instruction "pick up the cube" \
  --inject-franka-cameras \
  --enable_cameras \
  --robot-asset-name robot \
  --eef-body-name panda_hand
```
该命令会进入 IsaacLab 仿真环境，加载 `Isaac-Lift-Cube-Franka-v0` 任务，并使用指令 `"pick up the cube"` 开始调用 GR00T 推理服务进行控制。


### 3. 启动isaaclab的rokae+l10手组合
```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/load_xmate3_with_right_l10hand_urdf.py
```

打开手部摄像头的开关
```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/load_xmate3_with_right_l10hand_urdf.py --enable_cameras
```

### 4. 启动isaaclab的rokae+l10手数据采集的回放
```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/play_lerobot_rokae_xmate3_l10hand.py --enable_cameras
```

### 5. 启动ik库测试
```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/teleop_xmate3_l10hand_eef_ik.py --enable_cameras
```

### 6. 启动推理测试
```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/gr00t_xmate3_l10hand_cube_grasp.py --enable_cameras --policy-port 5555 --instruction "pick up the cube"
```

### 7. 启动训练
```bash
python examples/IsaacLab/finetune_l10_overfit.py \
  --instruction "pick up the bottle and place it in the box" \
  --max-steps 10000 \
  --global-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --tune-projector \
  --no-tune-diffusion-model \
  --no-tune-vlln \
  --load-bf16 \
  --gradient-checkpointing

python examples/IsaacLab/finetune_l10_overfit.py \
  --instruction "pick up the bottle and place it in the box" \
  --max-steps 10000 \
  --global-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --tune-projector \
  --no-tune-diffusion-model \
  --no-tune-vlln \
  --load-bf16 \
  --gradient-checkpointing \
  --use-swanlab \
  --swanlab-project rokae-xmate3-l10 
```

### 8. 推理0-shot
```bash
python examples/IsaacLab/compare_l10_gr00t_zero_shot_actions.py \
  --model-path checkpoints/rokae_xmate3_l10_overfit/checkpoint-5000 \
  --dataset-dir demo_data/l10_hand/lerobot_rokae_xmate3_linker_l10_groot_v1 \
  --modality-config-path examples/IsaacLab/rokae_xmate3_l10_modality_config.py \
  --instruction "pick up the bottle and place it in the box" \
  --episode-index 0 \
  --output-path outputs/IsaacLab/l10_overfit_compare_ep0.npz
```

### 9. 视频精修
```bash
pip install streamlit opencv-python pandas pyarrow
streamlit run examples/IsaacLab/trim_lerobot_episode_viewer.py
```

### 10. 使用精修后的trimmed数据集继续训练到20000轮
该脚本默认使用 `outputs/IsaacLab/trimmed_l10_dataset` 作为训练数据，输出目录仍然是 `checkpoints/rokae_xmate3_l10_overfit`。如果目录下已经有 `checkpoint-*`，会自动从数字最大的checkpoint继续训练，默认训练目标是 `20000` step。

```bash
python examples/IsaacLab/finetune_l10_overfit_on_trimmed_dataset.py
```

只准备并检查trimmed训练数据，不启动模型训练：

```bash
python examples/IsaacLab/finetune_l10_overfit_on_trimmed_dataset.py \
  --prepare-only
```

如果需要覆盖默认训练参数，可以继续追加和 `finetune_l10_overfit.py` 一样的参数：

```bash
python examples/IsaacLab/finetune_l10_overfit_on_trimmed_dataset.py \
  --max-steps 20000 \
  --global-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --use-swanlab \
  --swanlab-project rokae-xmate3-l10
```

### 11. 评估trimmed数据集上的L10过拟合模型
该脚本默认评估 `outputs/IsaacLab/trimmed_l10_dataset`，并自动选择 `checkpoints/rokae_xmate3_l10_overfit` 下数字最大的 `checkpoint-*`。评估结果会写入 `outputs/IsaacLab/l10_overfit_trimmed_eval`，包括 `evaluation_summary.json`、`per_episode_metrics.csv` 以及每个episode的预测对比文件。

```bash
python examples/IsaacLab/evaluate_l10_overfit_on_trimmed_dataset.py
```

指定使用 `checkpoint-20000` 进行评估：

```bash
python examples/IsaacLab/evaluate_l10_overfit_on_trimmed_dataset.py \
  --model-path checkpoints/rokae_xmate3_l10_overfit/checkpoint-20000
```

只评估部分episode并关闭图片绘制：

```bash
python examples/IsaacLab/evaluate_l10_overfit_on_trimmed_dataset.py \
  --model-path checkpoints/rokae_xmate3_l10_overfit/checkpoint-20000 \
  --episode-indices 0 1 2 \
  --no-plot
```


### 12.自己写的bench mark
Run with Isaac Lab's Python:
```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"   ~/Project/IsaacLab/isaaclab.sh -p examples/IsaacLab/evaluate_rokae_l10_bench_v0.py   --enable_cameras   --model-dir checkpoints/rokae_xmate3_l10_overfit   --dataset-dir outputs/IsaacLab/trimmed_l10_dataset   --num-trials 5  --image-height 800 --image-width 1200 --target-object bottle --cube-size 0.14 --cube-mass 0.06 --instruction "pick up the bottle and place it in the box" --cube-position=-0.7,-0.6,0.2
```