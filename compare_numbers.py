import csv
import math

def format_number(num):
    # Convert number to string with exactly one decimal place
    return f"{float(num):.1f}"

def is_similar(num1, num2):
    # Format both numbers to have exactly one decimal place
    str1 = format_number(num1)
    str2 = format_number(num2)
    
    # Pad with zeros to ensure both numbers have same length
    max_len = max(len(str1), len(str2))
    str1 = str1.zfill(max_len)
    str2 = str2.zfill(max_len)
    
    # Compare each digit
    for d1, d2 in zip(str1, str2):
        if d1 == '.' or d2 == '.':
            continue
        if abs(int(d1) - int(d2)) > 1:
            return False
    return True

def compare_csv_files(file1_path, file2_path):
    # Read numbers from first file
    numbers1 = []
    with open(file1_path, 'r', encoding='utf-8') as f1:
        csv_reader = csv.DictReader(f1)
        for row in csv_reader:
            numbers1.append(float(row['number']))
    
    # Read numbers from second file
    numbers2 = []
    with open(file2_path, 'r', encoding='utf-8') as f2:
        csv_reader = csv.DictReader(f2)
        for row in csv_reader:
            numbers2.append(float(row['number']))
    
    # Count similar numbers
    similar_count = 0
    total_count = min(len(numbers1), len(numbers2))
    
    for n1, n2 in zip(numbers1, numbers2):
        if is_similar(n1, n2):
            similar_count += 1
    
    return similar_count, total_count

if __name__ == "__main__":
    # Replace these with your actual file paths
    file1_path = "答案.csv"
    file2_path = "赛道1-张三-2013.csv"  # You need to specify the second file
    
    try:
        similar_count, total_count = compare_csv_files(file1_path, file2_path)
        print(f"近似数据的个数: {similar_count}")
        print(f"数据的总个数: {total_count}")
    except FileNotFoundError:
        print("Error: 请确保两个CSV文件都存在并且路径正确")
    except Exception as e:
        print(f"Error: {str(e)}") 