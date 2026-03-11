import torch as T
import torch.nn as nn

class RFF(nn.Linear):
    
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bandwidth: float = 10.,
            device: T.device | None = None,
            dtype: T.dtype | None = None,
        ):
        
        assert out_features % 2 == 0, f"{out_features=} is not divisible by 2!"
        
        super().__init__(in_features, out_features//2, False, device, dtype)
        self.bandwidth = bandwidth
        
        with T.no_grad():
            self.weight.data.normal_().mul_(bandwidth)
            self.weight.requires_grad_(False)
    
    def forward(self, x: T.Tensor) -> T.Tensor:
        
        x = super().forward(T.pi * x)
        return T.cat([x.sin(), x.cos()], dim=-1)
