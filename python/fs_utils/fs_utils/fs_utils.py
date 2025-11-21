import os

def make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Checked/created directory: {d}")
