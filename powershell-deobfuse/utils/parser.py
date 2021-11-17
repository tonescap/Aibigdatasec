import base64
import binascii

def parser(data: bytes) :#-> list[bytes]:
    # parse based on "()" and " "

    bracket_stack = 0
    single_comma_stack = 1
    double_comma_stack = 1

    parsed = []
    tmp = b''
    
    for i in range(len(data)):
        if data[i] == ord(b'('):
            bracket_stack += 1
            tmp += data[i].to_bytes(1, byteorder='little')

        elif data[i] == ord(b')'):
            bracket_stack -= 1
            tmp += data[i].to_bytes(1, byteorder='little')
        
        elif data[i] == ord(b'"'):
            double_comma_stack ^= 1
            tmp += data[i].to_bytes(1, byteorder='little')

        elif data[i] == ord(b"'"):
            single_comma_stack ^= 1
            tmp += data[i].to_bytes(1, byteorder='little')

        
        elif data[i] == ord(b' ') or data[i] == ord(b'\n') or data[i] == ord(b'\r') or data[i] == ord(b'|') or data[i] == ord(b';'):
            if bracket_stack == 0 and double_comma_stack and single_comma_stack:
                if tmp == b' ' or tmp == b'':
                    continue
                
                elif len(tmp) > 4 and tmp[-4:].lower() == b'join':
                    tmp += data[i].to_bytes(1, byteorder='little')
                    continue
                
                if tmp[-1] == ord(b'+') or  (tmp[-1] == ord(b'=') and not (len(tmp) > 2 and tmp[-2:] == b'==')) :
                    continue

                if len(data) != i+1 and (data[i+1] == ord(b'+') or data[i+1] == ord(b'=') ):
                    continue

                if len(data) > i + 6 and data[i+1:i+6].lower() == b'-join':
                    tmp += data[i].to_bytes(1, byteorder='little')
                    continue

                else:
                    if tmp[0] == ord(b'|'):
                        tmp = tmp[1:]
                        if tmp == b'':
                            continue
                    if tmp[-1] == ord(b'|'):
                        tmp = tmp[:-1]
                        if tmp == b'':
                            continue
                    
                    if tmp == b'&' or tmp == b'.':
                        tmp = b''
                        continue
                    
                    if len(tmp) >= 2 and tmp[:2] == b'&(':
                        tmp = tmp[1:]                  

                    parsed.append(tmp)
                    tmp = b''

            else:
                if tmp == b' ' or tmp == b'':
                    continue
                else:
                    tmp += data[i].to_bytes(1, byteorder='little')
        
        else:
            tmp += data[i].to_bytes(1, byteorder='little')
    
    if tmp != b'':
        parsed.append(tmp)
    return parsed
        

    




