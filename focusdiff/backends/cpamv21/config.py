import torch
class Config:
    SEED=10    
    
    SCALE_MASK=0.1
    THRES_HOLD=0.3
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    HEIGHT=768
    WIDTH=768
    MAX_STEP=50
    GUIDANCE_SCALE=7.5
    
    STEP_QUERY=7
    LAYER_QUERY=17
    
    STEP_CHANGE_MASK=1

class ConfigExpandMask(Config):
    SCALE_MASK=0.17
class ConfigSimilarObject(Config):
    STEP_QUERY=3
    LAYER_QUERY=7
    SCALE_MASK=0.01
class ConfigFixedMask(Config):
    SCALE_MASK=0.002
    
