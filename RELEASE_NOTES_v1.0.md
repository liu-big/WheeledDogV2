## WheeledDogV2 v1.0（稳定上楼梯）

自研轮足狗强化学习第一版可用基线。

### 亮点
- Isaac Lab / Isaac Sim 5.1 + RSL-RL rough / 楼梯课程
- MuJoCo sim2sim：15 cm / 20 cm 上下楼场景
- 预训练权重：`checkpoints/model_38700.pt`
- 作者：[liu-big](https://github.com/liu-big) · `2715209705@qq.com`

### 快速体验
```bash
git clone git@github.com:liu-big/WheeledDogV2.git
cd WheeledDogV2/sim2sim_wheeldog
python run_policy_wheeldog.py --scene 15up
```

详见仓库 [README](https://github.com/liu-big/WheeledDogV2#readme)。
