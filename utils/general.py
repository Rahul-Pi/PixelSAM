# The functions essential to run the detection algorithm

import torch

# Check if GPU is available
def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'SAM 🚀 torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    # print(s.encode().decode('ascii', 'ignore') if 'ascii' in s else s)  # emoji-safe

    return 'cuda' if cuda else 'cpu'
