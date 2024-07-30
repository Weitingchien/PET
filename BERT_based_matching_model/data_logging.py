import os

def write_samples_to_file(dataloader, file_path):
    # 收集所有樣本
    all_samples = []
    for batch in dataloader:
        all_samples.extend(batch)

    # 按照ID排序
    sorted_samples = sorted(all_samples, key=lambda x: x[0])  # x[0] 是 ID

    # 寫入排序後的樣本到iteration_i_samples.txt檔案

    with open(file_path, 'w', encoding='utf-8') as f:
        for id, text, hypothesis, binary_label, true_label in sorted_samples:
            f.write(f"ID: {id}\n")
            f.write(f"Text: {text}\n")
            f.write(f"Hypothesis: {hypothesis}\n")
            f.write(f"Binary Label: {binary_label}\n")
            f.write(f"True Label: {true_label}\n")
            f.write("\n")  # 添加空行空分隔不同樣本



def write_D_p_to_file(D_p, iteration, folder_name):
    sorted_D_p = sorted(D_p, key=lambda x: int(x[0]))
    file_path = os.path.join(folder_name, f'D_p_iteration_{iteration}.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for id, label, text in sorted_D_p:
            f.write(f"ID: {id}\n")
            f.write(f"Text: {text}\n")
            f.write(f"Label: {label}\n")
            f.write("\n")  # 添加空行以分隔不同的樣本