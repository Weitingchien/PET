# text_duplication_checker.py

import json
from collections import Counter
from typing import Dict, List, Tuple



def process_texts(texts: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """處理文本列表，返回所有文本的計數和重複文本的計數。"""
    text_counts = Counter(texts)
    duplicates = {text: count for text, count in text_counts.items() if count > 1}
    return dict(text_counts), duplicates

def check_text_duplication_tsv(file_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """檢查 TSV 格式文件中的第三個欄位（原始文本）是否有重複。"""
    original_texts = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                original_texts.append(parts[2])
            else:
                print(f"無效的行格式：{line}")

    return process_texts(original_texts)



def check_text_duplication_json(file_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    檢查指定文件中的 'original_text' 欄位是否有重複。

    Args:
    file_path (str): 要檢查的文件路徑。

    Returns:
    Tuple[Dict[str, int], Dict[str, int]]: 
        - 第一個字典包含所有文本及其出現次數。
        - 第二個字典只包含重複的文本及其出現次數。
    """
    original_texts = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # 將每行解析為 JSON
                data = json.loads(line)
                # 從 JSON 中提取 'original_text' 並添加到列表中
                original_texts.append(data['original_text'])
            except json.JSONDecodeError:
                print(f"解析 JSON 時出錯，行內容：{line}")
    # 使用 Counter 計算每個文本的出現次數
    text_counts = Counter(original_texts)
    # 創建一個新字典，只包含出現次數大於 1 的文本
    duplicates = {text: count for text, count in text_counts.items() if count > 1}
    # 返回所有文本的計數和重複文本的計數
    return process_texts(original_texts)



def print_duplication_report(text_counts: Dict[str, int], duplicates: Dict[str, int]) -> None:
    """
    打印文本重複的報告。

    Args:
    text_counts (Dict[str, int]): 所有文本及其出現次數。
    duplicates (Dict[str, int]): 重複的文本及其出現次數。
    """
    if duplicates:
        print("發現重複的 original_text:")
        for text, count in duplicates.items():
            print(f"文本 '{text}' 出現了 {count} 次")
    else:
        print("沒有發現重複的 original_text")

    print(f"\n總共有 {len(text_counts)} 個 original_text")
    print(f"其中有 {len(duplicates)} 個是重複的")
    print(f"重複的 original_text 總數：{sum(duplicates.values())}")


def normalize_text(text):
        """將文本標準化，移除標點符號和空格，轉為小寫"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # 移除標點符號
        text = re.sub(r'\s+', '', text)  # 移除所有空白字符
        return text

def remove_duplicates_and_write(input_file: str, output_file: str) -> None:
    """
    讀取TSV文件, 移除重複的原始文本, 並將結果寫入新文件
    """
    seen_texts = set()
    removed_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                original_text = parts[2]
                normalized_text = normalize_text(original_text)

                if normalized_text not in seen_texts:
                    seen_texts.add(normalized_text)
                    outfile.write(line)
                else:
                    removed_count += 1
            else:
                print(f"無效的行格式：{line}")

    print(f"已從 {input_file} 中移除 {removed_count} 個重複項")
    print(f"去重後的數據已寫入 {output_file}")















def main():
    # 處理 sorted_confidence_samples.txt
    """
    json_file_path = './BERT_self_training_0.8/sorted_confidence_samples.txt'
    print("檢查 sorted_confidence_samples.txt:")
    text_counts_json, duplicates_json = check_text_duplication_json(json_file_path)
    print_duplication_report(text_counts_json, duplicates_json)
    """

    # 處理 train_pu_half_v1.txt
    tsv_file_path = './datasets/emotion/train_pu_half_v1.txt'
    print("\n檢查 train_pu_half_v1.txt:")
    text_counts_tsv, duplicates_tsv = check_text_duplication_tsv(tsv_file_path)
    print_duplication_report(text_counts_tsv, duplicates_tsv)
    

    # 移除重複的原始文本並寫入新文件
    
    output_file_path = './datasets/emotion/train_pu_half_v1_deduplication.txt'
    remove_duplicates_and_write(tsv_file_path, output_file_path)
    

    tsv_file_path = './datasets/emotion/train_pu_half_v1_deduplication.txt'
    text_counts_tsv, duplicates_tsv = check_text_duplication_tsv(tsv_file_path)
    print_duplication_report(text_counts_tsv, duplicates_tsv)

if __name__ == "__main__":
    main()