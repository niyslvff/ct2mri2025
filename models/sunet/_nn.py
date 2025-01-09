from torch import nn

class _nn:
    @staticmethod
    def conv_nd(dims, *args, **kwargs):
        if dims == 1:
            return nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            return nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            return nn.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    @staticmethod
    def batch_norm_nd(dims, *args, **kwargs):
        if dims == 2:
            return nn.BatchNorm2d(*args, **kwargs)
        elif dims == 3:
            return nn.BatchNorm3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")

    @staticmethod
    def avg_pool_nd(dims, *args, **kwargs):
        if dims == 1:
            return nn.AvgPool1d(*args, **kwargs)
        elif dims == 2:
            return nn.AvgPool2d(*args, **kwargs)
        elif dims == 3:
            return nn.AvgPool3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")
