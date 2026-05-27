import torch
class Config:
    SEED=10
    SCALE_MASK=0.1
    THRES_HOLD=0.25
    DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
    HEIGHT=512
    WIDTH=512
    MAX_STEP=50
    GUIDANCE_SCALE=10
    STEP_QUERY=7
    LAYER_QUERY=16
