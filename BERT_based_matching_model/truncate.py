import os

def truncate_file(filename, max_size_mb):
    max_size_bytes = max_size_mb * 1024 * 1024  
    
    if os.path.getsize(filename) <= max_size_bytes:
        print("文件大小已經小於或等於指定大小")
        return
    
    with open(filename, 'rb+') as file:
        file.seek(max_size_bytes)
        file.truncate()




if __name__ ==  "__main__":
    truncate_file('./test_pred.txt', 28)