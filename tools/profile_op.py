import torch

from fse_ops import median_filter5x5

SHAPE = (4, 3, 512, 512)
REPEATS = 5
THREADS = 1

torch.manual_seed(0)
torch.set_num_threads(THREADS)
x = torch.randn(SHAPE, dtype=torch.float32)

for _ in range(REPEATS):
    result = median_filter5x5(x)

print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(result.shape))
print("Calls:", REPEATS)
print("Threads:", THREADS)
