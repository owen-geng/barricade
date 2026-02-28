import torch
print(torch.cuda.is_available())
print(torch.accelerator.is_available())
print(torch.accelerator.current_accelerator().type)