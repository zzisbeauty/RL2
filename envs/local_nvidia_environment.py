import re  
from typing import Dict  
  
async def step(state: str, action: str, extra_info: Dict) -> Dict:  
    """  
    问答环境：基于参考答案评估模型回答  
    """  
    reference = extra_info.get("reference_answer", "").lower()  
    response = action.lower()  
      
    # 简单的关键词匹配奖励  
    reward = 0.0  
    score = 0.0  
      
    if reference:  
        # 计算词汇重叠度  
        ref_words = set(reference.split())  
        action_words = set(response.split())  
          
        if ref_words:  
            overlap = len(ref_words & action_words) / len(ref_words)  
            reward = overlap  
            score = overlap  
      
    return {  
        "next_state": None,  
        "reward": float(reward),  
        "score": float(score),  
        "done": True,  
        "extra_info": extra_info  
    }