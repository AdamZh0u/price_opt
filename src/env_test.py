import os
from src import const
print(os.environ['PYTHONPATH'])
print(os.environ['PATH'])
print(const.ROOT)
print(os.environ['ENV_PARM']) # params in .env file
