import os
import sys
import subprocess


self_folder = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "--no-warn-script-location"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-r", f"{self_folder}/requirements.txt", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "--no-warn-script-location"])