import zlib
import gzip
import base64

def decode_base64_and_inflate( b64string ):
    decoded_data = base64.b64decode( b64string )
    return zlib.decompress( decoded_data , -15)

def deflate_and_base64_encode( string_val ):
    zlibbed_str = zlib.compress( string_val )
    compressed_string = zlibbed_str[2:-4]
    return base64.b64encode( compressed_string )


def decode_base64_and_gzip( b64string ):
    decoded_data = base64.b64decode( b64string )
    return gzip.decompress( decoded_data )

if __name__ == "__main__":
    import os

    dirlist = os.listdir("./phase0_v2")

    cnt = 0

    def file_decode(data):

        if b"frombase64string('" in data.lower() and len(data.split(b"\n")) < 10:
            #print(data[:200])
            idx = data.lower().index(b"frombase64string('")
            idx = idx + len("frombase64string' ")
            data = data[idx:]
            idx = data.index(b"'")
            #print(data[:idx])
            global cnt
            cnt += 1
            return (decode_base64_and_inflate(data[:idx]))
        return data

    for file in dirlist:
        #print(file)
        with open(f"./phase0_v2/{file}", "rb") as f:
            data = f.read()
        
        data = file_decode(data)

        with open(f"./phase0_v2_decode/{file}", "wb") as f:
            f.write(data)
            f.close()

    print(cnt)