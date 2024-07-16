import random

def extract_samples(input_file, output_file, target_labels):
    extracted_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                label, source, text = parts[0], parts[1], '\t'.join(parts[2:])
                if label in target_labels:
                    extracted_samples.append(f"{label}\t{source}\t{text}\n")
            else:
                print(f"Warning: Skipping malformed line: {line}")
    
    # 可選:打亂樣本順序
    # random.shuffle(extracted_samples)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(extracted_samples)
    
    print(f"Extracted {len(extracted_samples)} samples to {output_file}")




def main():
    # 設置輸入和輸出文件路徑
    """
    test_input_file = './datasets/emotion/test.txt'  # 測試集文件路徑
    unseen_test_I_output_file = './datasets/emotion/unseen_class_test_I.txt'  # unseen測試集

    # 設置預提取的標籤
    unseen_test_I_target_labels = ['joy', 'guilt', 'disgust', 'surprise']

    # 執行提取
    extract_samples(test_input_file, unseen_test_I_output_file, unseen_test_I_target_labels)
    """

    dev_input_file = './datasets/emotion/dev.txt'
    dev_v0_output_file = './datasets/emotion/dev_v0.txt'
    dev_v1_output_file = './datasets/emotion/dev_v1.txt'

    dev_v1_target_labels = ['joy', 'guilt', 'disgust', 'surprise']
    extract_samples(dev_input_file, dev_v1_output_file, dev_v1_target_labels)

if __name__ == "__main__":
    main()