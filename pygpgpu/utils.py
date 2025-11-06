import hashlib


def md5(content:str)->str:
    byte_string = content.encode('utf-8')
    md5_hash = hashlib.md5()
    md5_hash.update(byte_string)
    
    return md5_hash.hexdigest()