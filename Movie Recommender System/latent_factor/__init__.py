import os
from .config import *
from .preprocessor import *
from .model import *

if not os.path.exists('./latent_factor/binaries'):
    os.makedirs("./latent_factor/binaries")
