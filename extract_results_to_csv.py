import os
import csv
import ast




def extract_results_to_csv(root_dir, output_file, header_written):
    results = []
    print(f'root_dir: {root_dir}')
    for folder in os.listdir(root_dir):
        if root_dir.startswith("./output"):
            output_folder = os.path.join(root_dir, folder)
            for sub_folder in os.listdir(output_folder):
                print(f'sub_folder: {sub_folder} output_folder: {output_folder}')
                if output_folder.endswith(("yelp", "yelp-i1", "yelp-i2", "yelp-i2-distilled")):
                    if os.path.isdir(os.path.join(output_folder ,sub_folder)):
                        yelp_folder = os.path.join(output_folder, sub_folder)
                        print(f'yelp_folder: {yelp_folder}')
                        for pi_folder in os.listdir(yelp_folder):
                            pi_path = os.path.join(yelp_folder, pi_folder)
                            print(f'pi_path: {pi_path}')
                            if pi_path.endswith('results.txt'): 
                                results_file = pi_path
                                if os.path.exists(results_file):
                                    with open(results_file, "r") as file:
                                        content = file.read()
                                        try:
                                            data = ast.literal_eval(content)
                                            print(f'data: {data}')
                                            row = [
                                                yelp_folder,
                                                data.get("train_set_before_training", ""),
                                                data.get("train_set_after_training", ""),
                                                data.get("test_set_after_training", "")
                                            ]
                                            results.append(row)
                                            print(f'results: {results}')
                                        except (SyntaxError, ValueError):
                                            print(f"Invalid content in {pi_path}")
    # 對results列表進行排序
    results.sort(key=lambda x: x[0])
    
    with open(output_file, "a", newline="") as file:
        writer = csv.writer(file)
        if not header_written:
            writer.writerow(["output_folder", "train_set_before_training", "train_set_after_training", "test_set_after_training"])
        
        writer.writerows(results)

    print(f"Results exported to {output_file}")

def main():
    root_dir = ["./output_supervised_roberta_50", "./output_supervised_roberta_100", "./output_supervised_roberta_1000", "./output_supervised_roberta_10000", "./output_roberta_iPET_100"]
    output_file = "results.csv"
    header_written = False  # 標記列名是否已經寫入
    for r_dir in root_dir:
        extract_results_to_csv(r_dir, output_file, header_written)
        header_written = True



if __name__ == "__main__":
    main()