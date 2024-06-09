import os
import re
import matplotlib.pyplot as plt



def main():
    # 定義要讀取的資料夾路徑
    folders = [
        "./output_supervised_roberta_50/yelp",
        "./output_supervised_roberta_10000/yelp",
        "./output_supervised_roberta_1000/yelp",
        "./output_supervised_roberta_100/yelp",
        "./output_final_roberta/yelp",
        "./output_roberta_iPET_100/yelp-i2-distilled"
    ]

    pattern_scores = {}
    all_scores = []

    for folder in folders:
        result_file = os.path.join(folder, "result_test.txt")
        with open(result_file, "r") as f:
            content = f.read()

        # 使用正則表達式提取pattern分數
        pattern_scores[folder] = {}
        matches = re.findall(r"acc-p(\d+): ([\d.]+) \+- ([\d.]+)", content)
        for match in matches:
            pattern = int(match[0])
            mean = float(match[1])
            std = float(match[2])
            pattern_scores[folder][pattern] = (mean, std)

        # 提取all分數
        match = re.search(r"acc-all-p: ([\d.]+) \+- ([\d.]+)", content)
        if match:
            mean = float(match.group(1))
            std = float(match.group(2))
            all_scores.append((folder, mean, std))
    # 繪製pattern平均分數圖

    task = ["supervised(50)", "supervised(10000)", "supervised(1000)", "supervised(100)", "PET", "iPET"]
    i = 0
    fig, ax = plt.subplots(figsize=(10, 6))
    for folder, scores in pattern_scores.items():
        patterns = sorted(scores.keys())
        means = [scores[p][0] for p in patterns]
        plt.plot(patterns, means, marker="o", label=task[i])
        # line, = ax.plot(patterns, means, marker="o", label=folder)

        # 添加數值標註
        for x, y in zip(patterns, means):
            ax.annotate(f"{y:.4f}", (x, y), xytext=(5, 5), textcoords="offset points")
        i += 1
    plt.xlabel("Pattern")
    plt.ylabel("Accuracy")
    plt.title("Pattern Average Scores(Pretrained language model: roberta-base)")
    plt.xticks(range(4)) # 設置x軸刻度為0到3
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()