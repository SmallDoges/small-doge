import datasets
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import os

def convert_parquet_to_dataset():
    # 输入和输出路径
    input_path = '/path/to/training_data/data/train-00000-of-00001.parquet'
    output_dir = '/path/to/training_data/processed_dataset'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("Reading parquet file...")
    # 使用pandas读取parquet文件
    df = pd.read_parquet(input_path)
    
    print("Converting to datasets format...")
    # 将pandas DataFrame转换为Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # 显示数据集信息
    print("\nDataset Info:")
    print(dataset)
    
    print(f"\nSaving dataset to {output_dir}...")
    # 保存数据集
    dataset.save_to_disk(output_dir)
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    convert_parquet_to_dataset()