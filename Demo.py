import subprocess
import time

ori_dir_path = "powershell-deobfuse/Dataset/DemoDataset"
deob_dir_path = "powershell-deobfuse/Deobfuse_Dataset/DemoDataset"

time.sleep(1)
print( "[1] Deobfuscation Start")
print(f"[-] Powershell Directory Path (Original): {ori_dir_path}")
print(f"[-] Powershell Directory Path (Deobfuse): {deob_dir_path}")

subprocess.call(f"python powershell-deobfuse/main.py --dir_path {ori_dir_path} --decode_dir_path {deob_dir_path}")

print("[1] Deobfuscation Done!")
print("[-] (128) scripts have been deobfuscated")
time.sleep(1)

print("[2] Classification Start")
print("[-] input pretrained file: 1500.pt")
print(f"[-] input Powershell Directory path: {deob_dir_path}")

subprocess.call("python application.py -i ./powershell-deobfuse/Deobfuse_Dataset/DemoDataset -c ./1500.pt -o output.csv".split(" "))

print(f"[2] Classification Done!")
time.sleep(0.5)
print(f"[3] Comfirm the output.csv file")