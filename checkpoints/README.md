# Checkpoints

| 文件 | 说明 |
|------|------|
| [`model_38700.pt`](model_38700.pt) | **v1.0 推荐权重**：Rough 楼梯课程约 38700 iter；MuJoCo 15/20 cm 上下楼可用 |

加载示例：

```bash
cd sim2sim_wheeldog
python run_policy_wheeldog.py --scene 15up --ckpt ../checkpoints/model_38700.pt
```

Isaac play（需本机有 Isaac Lab）：

```bash
CKPT=model_38700.pt RUN_DIR=../checkpoints bash play_wheeled_dog_v2.sh
# 或将权重拷入 rl_training/logs/rsl_rl/wheeled_dog_v2_rough/<run>/ 后按 play 脚本默认路径加载
```
