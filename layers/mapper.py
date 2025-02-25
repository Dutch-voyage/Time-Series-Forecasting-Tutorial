import torch
import torch.nn as nn
import torch.nn.functional as F

class Mapper(nn.Module):
    def __init__(self,
                 aux_model: nn.Module,
                 task: str,
                 aux_model_stu: nn.Module = None,):
        super(Mapper, self).__init__()
        self.auxmodel = aux_model
        # self.aux_model_stu = aux_model_stu(task=task)
        if aux_model_stu is not None:
            self.auxmodelstu = aux_model_stu
        else:
            self.auxmodelstu = None

    def forward(self, x, y):
        result = self.auxmodel(x, y, None, None)
        if self.auxmodelstu is not None:
            result_stu = self.auxmodelstu(x.detach(), y.detach(), None, None)
        else:
            result_stu = None
        return result