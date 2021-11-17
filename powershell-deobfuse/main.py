import base64
import os
import shutil
import subprocess as sp
import argparse
import binascii
import unicodedata

from utils.base64_decode import decode_base64_and_gzip, decode_base64_and_inflate
import utils.logger
from utils.parser import parser

cnt = 0
end = 10000
logger = utils.logger.make_logger()

def NeedDeob(data: bytes, file : str) -> bool:
    data_1 = data.split(b"|")
    data_2 = data.split(b"\n")
    data_3 = data.split(b';')

    if (len(data_1) < 5 and len(data_2) < 10) or (len(data_3) >= 8 and len(data_3) <= 20  and len(data_2) < 10):
        if len(data) > 1024 * 1024:
            if b'frombase64string(' in data.lower() and b'deflatestream(' in data.lower():
                logger.critical(f"file {file} | Too Big file size {len(data)} | base 64 ok!")
                return 2
            elif b'frombase64string(' in data.lower() and b'gzipstream(' in data.lower():
                logger.critical(f"file {file} | Too Big file size {len(data)} | base 64 and gzip ok!")
                return 3
            else:
                logger.error(f"file {file} | Too Big file size {len(data)}")
                return 0
        return 1
    else:
        return 0
        '''
        Since It's hard to paring....

        if b'frombase64string(' in data.lower() and b'deflatestream(' in data.lower():
            return 2
            
        elif b'frombase64string(' in data.lower() and b'gzipstream(' in data.lower():
            return 3

        else:
            return 0
        '''


def Deobfuse(data : bytes, file : str):
    try:
        p = sp.Popen(f"powershell echo {data.decode()}".split(' '), stdout=sp.PIPE, stderr=sp.PIPE)
    except FileNotFoundError as e:
        logger.warning(f'Error on {file} | error msg: FileNotFoundError-{e}')
        with open("./tmp.ps1", "wb") as f:
            f.write(b'echo ' + data)
            f.close()
        p = sp.Popen(f"powershell ./tmp.ps1".split(' '), stdout=sp.PIPE, stderr=sp.PIPE)
    except ValueError as e:
        logger.warning(f'Error on {file} | error msg: ValueError-{e}')
        return b'\r\n'

    out, err = p.communicate()
    rc = p.returncode
    
    
    if DEBUG: 
        if len(data.decode()) > 90:
            logger.info(f'file {file} | return code: {rc} | data: {data.decode()[:90]}    ...    {data.decode()[-30:]}')
        else:
            logger.info(f'file {file} | return code: {rc} | data: {data}')

    if rc == 0:
        if DEBUG:
            if len(out) > 100:
                logger.info(f'stdout : {out[:70]}    ...   {out[-20:]} ')
            else:
                logger.info(f'stdout : {out[:100]}')

        if len(out.split(b'\n')) > 200 and (len(out)) < len(out.split(b"\n")) * 3 + 20:
            logger.warning(f'Error on {file} | error msg: too many enter')
            
            with open("./tmp.ps1", "wb") as f:
                f.write(data)
                f.close()
            p = sp.Popen(f"powershell ./tmp.ps1".split(' '), stdout=sp.PIPE, stderr=sp.PIPE)

            out, err = p.communicate()
            rc = p.returncode


    else:
        logger.warning(f'Error on {file} | error msg: {err[:100]}')
        if b'This script contains malicious' in err:
            logger.error("RealTime AV is working! Off it")


    if len(out) > 1000:
        try:
            tmp = out.strip()
            tmp = base64.b64decode(tmp, validate=True)
            logger.info(f'base64 decode on file {file}')

            is_utf16 = True
            for i in range(10):
                if tmp[2*i+1] != 0:
                    is_utf16 = False
            
            if is_utf16:
                tmp = tmp.decode('utf-16').encode()
            
            tmp += b"\r\n"

            out = tmp
        except binascii.Error as e:
            pass


    return out

def b64_deob_inflate(data : bytes):
    Wrong = False

    start= data.lower().index(b'frombase64string(') + 17
    _data = data[start:]
    try:
        _data = _data[_data.index(b"'")+1:]
        end= _data.index(b"'")
    except ValueError as e:
        try:
            _data = _data[_data.index(b'"')+1:]
            end= _data.index(b'"')
        except ValueError as e:
            Wrong = True
            logger.error(f"file {file} | b64_inflate_deobfuscation Error")

    if Wrong: 
        return data

    _data = _data[:end]

    return decode_base64_and_inflate(_data)


def b64_deob_gzip(data : bytes):
    Wrong = False

    start= data.lower().index(b'frombase64string(') + 17
    _data = data[start:]
    try:
        _data = _data[_data.index(b"'")+1:]
        end= _data.index(b"'")
    except ValueError as e:
        try:
            _data = _data[_data.index(b'"')+1:]
            end= _data.index(b'"')
        except ValueError as e:
            Wrong = True
            logger.error(f"file {file} | b64_gzip_deobfuscation Error")

    if Wrong: 
        return data

    _data = _data[:end]

    return decode_base64_and_gzip(_data)




if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="PowerShell Deobfusecator")

    argparser.add_argument('--dir_path',           type=str, default="./test_ps1", help='malicious powershell directory path')
    argparser.add_argument('--decode_dir_path',    type=str, default="./tmp",      help='decoded powershell directory path (will be decoded on this folder)')
    argparser.add_argument('--logfile',            type=str, default="test.log",   help='logging file name (default : test.log)')
    argparser.add_argument('--PASS',               type=bool, default=True,        help='pass the exist files or not')
    argparser.add_argument('--debug',              type=bool, default=False,       help='debug message')

    args = argparser.parse_args()

    #################### configuration #####################

    #dir_path = "./phase0_v2"
    #dir_path = "./test_ps1"
    #dir_path = "./decoded_ps1"
    dir_path = args.dir_path

    #decode_dir_path = "./decoded_ps1"
    #decode_dir_path = "./decoded_ps2"
    #decode_dir_path = "./tmp"
    decode_dir_path = args.decode_dir_path

    DEBUG = args.debug
    logger = utils.logger.make_logger(logfile=args.logfile)
    PASS = args.PASS
    
    dirlist = os.listdir(dir_path)
    dirlist.sort()

    ########################################################


    for file in dirlist:
        if PASS:
            if os.path.exists(f"{decode_dir_path}/{file}"):
                continue


        with open(f"{dir_path}/{file}", "rb") as f:
            data = f.read()
        
        need_deob = NeedDeob(data, file)
        if need_deob == 1:

            f = open(f'{decode_dir_path}/{file}', "wb")

            data = parser(data)
            try:
                for d in data: 
                    decoded = Deobfuse(d, file)
                    f.write(decoded)
                
                f.close()
            except KeyboardInterrupt as e:
                print(f"KeyboardInterrupt at {file}")
                exit(1)

        elif need_deob == 2:

            f = open(f'{decode_dir_path}/{file}', "wb")
            
            decoded = b64_deob_inflate(data)
            f.write(decoded)
        
            f.close()
        
        elif need_deob == 3:

            f = open(f'{decode_dir_path}/{file}', "wb")
            
            decoded = b64_deob_gzip(data)
            f.write(decoded)
        
            f.close()

        else:
            #shutil.copy(f'{dir_path}/{file}', f'{decode_dir_path}/{file}')
            pass
                
        
        if cnt % 10 == 0:
            logger.info(f'cnt : {cnt}')
        cnt += 1
        if cnt == end:
            break