# groot-isaaclab-simulator

这个项目用于在 IsaacLab 环境中调用 GR00T 推理模型，让机器人根据自然语言指令执行任务。

## 快速启动

建议分别打开两个终端：一个终端启动 GR00T 推理服务，另一个终端启动 IsaacLab 仿真并连接推理服务。

### 1. 启动推理模型

在项目根目录下运行：

```bash
python ./gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/GR00T-N1.7-3B \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --port 5555
```

该命令会加载 `checkpoints/GR00T-N1.7-3B` 模型，并在 `5555` 端口启动推理服务。

### 2. 启动 IsaacLab 并开始推理

推理服务启动后，在另一个终端运行：

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
