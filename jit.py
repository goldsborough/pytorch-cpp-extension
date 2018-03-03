import torch.utils.cpp_extension
lltm = torch.utils.cpp_extension.load(
    name="lltm", sources=["lltm.cpp"], verbose=True)
help(lltm)
