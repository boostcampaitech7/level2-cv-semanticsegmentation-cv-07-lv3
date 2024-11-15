import os
import pandas as pd

import importlib
import subprocess

package_name = "openpyxl"

try:
    importlib.import_module(package_name)
except ImportError:
    subprocess.check_call(["pip", "install", package_name])
    
def excel_to_csv(excel_file_path, sheet_name=0, csv_file_path=None):
    
    print("현재 디렉터리:", os.getcwd())

    try:
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    except FileNotFoundError as e:
        print(f"지정 경로에 엑셀 파일이 없음: {excel_file_path}")
        raise e

    # 특수 문자 `_x0008_` 제거
    df = df.map(lambda x: x.replace('_x0008_', '') if isinstance(x, str) else x)
    df = df.map(lambda x: x.replace(' ', '') if isinstance(x, str) else x)

    # CSV 파일 경로 설정 (경로가 제공되지 않을 경우 기본 경로 사용)
    if csv_file_path is None:
        csv_file_path = excel_file_path.replace('.xlsx', '.csv')

    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

    print(f"저장 완료: {csv_file_path}")


excel_file_path = "data/meta_data.xlsx"
csv_file_path = "data/meta_data.csv"
excel_to_csv(excel_file_path, csv_file_path=csv_file_path)
