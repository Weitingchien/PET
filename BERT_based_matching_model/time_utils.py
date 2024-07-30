# time_utils.py
import os
import csv

def write_time_info(folder_name, total_time, num_iterations):
    total_time_hours = total_time / 3600  # Convert seconds to hours
    avg_time_hours = total_time_hours / num_iterations

    csv_path = os.path.join(folder_name, 'time_info.csv')
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Total Time (hours)", "Number of Iterations", "Average Time per Iteration (hours)"])
        writer.writerow([f"{total_time_hours:.2f}", num_iterations, f"{avg_time_hours:.2f}"])