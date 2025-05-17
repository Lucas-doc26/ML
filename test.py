from utils.datasets import *
from pathlib import Path


download_all_datasets(Path('datasets'))
create_x_ray_dataset(path_dataset=Path('datasets/chest_xray'))
create_batches(['X-Ray'])