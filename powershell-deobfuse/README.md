# PowerShell-Deobfuse
PowerShell Deobfusecator

## How to use

```shell
>python ./main.py
usage: main.py [-h] [--dir_path DIR_PATH] [--decode_dir_path DECODE_DIR_PATH] [--logfile LOGFILE]
               [--PASS PASS] [--debug DEBUG]

PowerShell Deobfusecator

options:
  -h, --help            show this help message and exit
  --dir_path DIR_PATH   malicious powershell directory path
  --decode_dir_path DECODE_DIR_PATH
                        decoded powershell directory path (will be decoded on this folder)
  --logfile LOGFILE     logging file name (default : test.log)
  --PASS PASS           pass the exist files or not
  --debug DEBUG         debug message
```
