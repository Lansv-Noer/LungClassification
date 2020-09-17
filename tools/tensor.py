import torch
import numpy as np

def onehot(input: torch.Tensor, dim: int, n_classes: int):
    return torch.zeros([input.shape[0], n_classes, *input.shape[1:]], device=input.device).scatter_(
        dim=dim, index=input.unsqueeze(dim), src=torch.tensor(1))


def tensor2img(input: torch.Tensor):
    if input.device.type == "cuda":
        if input.ndim == 4:
            tensor = input.permute([0,2,3,1])
            result = []
            for ten in tensor:
                result.append(ten.cpu().numpy())
            return result
        elif input.ndim == 3:
            return input.cpu().numpy()
    else:
        if input.ndim == 4:
            tensor = input.permute([0,2,3,1])
            result = []
            for ten in tensor:
                result.append(ten.numpy())
            return result
        elif input.ndim == 3:
            return input.numpy()

