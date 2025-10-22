import os
from datetime import timedelta
import torch
import torch.distributed as dist

def initialize_global_process_group(timeout_second=36000):
    # 单卡模式使用 gloo 后端,避免 NCCL 网络配置问题   × 错误的设置

    # dist.init_process_group("gloo", timeout=timedelta(seconds=timeout_second)) # 单卡，使用 gloo backend
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second)) # original 多卡通信
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def _unwrap_process_group(process_group):
    if hasattr(process_group, '_wrapped'):
        process_group = process_group._wrapped
    elif hasattr(process_group, 'group'):
        process_group = process_group.group
    return process_group

def broadcast_object(obj, src=None, group=None, group_src=None):
    object_list = [obj]
    dist.broadcast_object_list(
        object_list,
        src=src,
        group=_unwrap_process_group(group),
        group_src=group_src
    )
    return object_list[0]

def gather_and_concat_list(lst, process_group):

    process_group = _unwrap_process_group(process_group)
    lists = (
        dist.get_world_size(process_group) * [None]
        if dist.get_rank(process_group) == 0
        else None
    )
    dist.gather_object(
        lst,
        lists,
        group=process_group,
        group_dst=0
    )
    return sum(lists, []) if dist.get_rank(process_group) == 0 else None
