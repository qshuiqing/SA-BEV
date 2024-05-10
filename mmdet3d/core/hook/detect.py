import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DetectAnomalyHook(Hook):

    def before_run(self, runner):
        print(r'detect anomaly open.!')
        torch.autograd.set_detect_anomaly(True)
