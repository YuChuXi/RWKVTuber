import subprocess

dataset = "sample"
subprocess.run(["./main", "-m","models/ggml-large-v3.bin", "-f", "\"dataset/{dataset}/raw/1.mp4\""], shell=True)