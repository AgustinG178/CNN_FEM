import math
from torch.optim.lr_scheduler import _LRScheduler

class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay Scheduler.
    Muy efectivo para datasets grandes para asegurar que el modelo explore 
    el espacio de parámetros a LR alto antes de colapsar al mínimo.
    """
    def __init__(self, optimizer, warmup_steps, stable_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WSDScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        
        # 1. Warmup (Linear)
        if step < self.warmup_steps:
            factor = step / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
            
        # 2. Stable (Constant)
        if step < self.stable_steps:
            return [base_lr for base_lr in self.base_lrs]
            
        # 3. Decay (Cosine)
        decay_step = step - self.stable_steps
        decay_total = self.total_steps - self.stable_steps
        factor = 0.5 * (1 + math.cos(math.pi * decay_step / decay_total))
        
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]
