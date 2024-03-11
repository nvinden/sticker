from urllib.request import urlretrieve
from json import load
from PIL import Image
from os.path import join
import pandas as pd

import numpy as np
import shutil

# Example to get data from part-000001
# Download part-000001.zip
for part_id in range(1, 2 + 1):
	part_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-{part_id:06}.zip'
	urlretrieve(part_url, f'data/raw/part-{part_id:06}.zip')

#import pandas as pd
df = pd.read_parquet('/home/nvinden/Work/bruce/sticker/data/raw/metadata.parquet')
df.to_csv('/home/nvinden/Work/bruce/sticker/data/raw/metadata.csv')
