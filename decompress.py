import tarfile
import glob



def find_tar_gz():
    datasets = glob.glob("./text_classification_datasets/*tar.gz")
    return datasets


# 這個函數將解壓縮指定的 .tar.gz 檔案到當前目錄或指定目錄
def decompress_tar_gz(file_path, output_path='./text_classification_datasets'):
    # 檢查檔案是否為 .tar.gz 格式
    if file_path.endswith('.tar.gz'):
        # 開啟壓縮檔案
        with tarfile.open(file_path, 'r:gz') as tar:
            # 解壓縮
            tar.extractall(path=output_path)
            print(f'檔案已解壓縮到 {output_path}')
    else:
        print('這不是一個 .tar.gz 檔案')

# 使用函數
# 替換 'your_file_path_here.tar.gz' 為你的 .tar.gz 檔案路徑
# 如果您想要解壓到特定目錄，也替換 'your_output_path_here'
# decompress_tar_gz('your_file_path_here.tar.gz', 'your_output_path_here')



def main():
    text_classification_datasets = find_tar_gz()
    # print(f'text_classification_datasets:{text_classification_datasets}')
    for f in text_classification_datasets:
        print(f)
        decompress_tar_gz(f)
    # decompress_tar_gz()


if __name__ == "__main__":
    main()