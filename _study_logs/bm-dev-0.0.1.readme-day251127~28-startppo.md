## 分支 / 工作说明

此分支学习 PPO 相关内容

### _todo list_

- [ ] 尝试运行 PPO，添加了配置等信息，但是这个项目的运行很复杂，不清晰。因此使用这个项目了。



## PPO 训练代码入口

`RL2/trainer/ppo.py` debug with ppo launch json config settings



## PPO 代码细节

### _model train 配置文件_

`RL2/trainer/config/local_nvidia_ppo.yml`

- model train config 文件中的字段解释 https://deepwiki.com/search/-ppo_55099ad6-7cc5-41b1-8cac-d9c70626db3d?mode=fast#11

- 配置文件中的 Rollout 系统和 fsdp 参数解释：https://deepwiki.com/search/-ppo_55099ad6-7cc5-41b1-8cac-d9c70626db3d?mode=fast#13


### _model train env 配置文件_

`envs/local_nvidia_environment.py`

这里说明了 env 这个文件的作用，很清晰，直观的展示了 RL 的本质工作路径：https://deepwiki.com/search/-ppo_55099ad6-7cc5-41b1-8cac-d9c70626db3d?mode=fast#12

可以看出来，这个配置文件非常重要

![](./assets/20251128093743.png)


