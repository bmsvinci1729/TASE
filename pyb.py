# find_data_path.py
import pybullet_data
import os

data_path = pybullet_data.getDataPath()
print(f"[LOG] PyBullet data path is: {data_path}")
# On a typical Linux system, this might print something like:
# /home/user/TASE/venv/lib/python3.10/site-packages/pybullet_data