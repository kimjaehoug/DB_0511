import pandas as pd
import os
import glob
import pymysql
import urllib.parse

# -------------------------- MySQL 연결 설정 --------------------------
user = 'famers'
password = urllib.parse.quote_plus('1633')
host = 'localhost'
port = '3306'
database = 'famers'
table_name = 'EDAdata'

conn = pymysql.connect(
    host=host,
    port=int(port),
    user=user,
    password=urllib.parse.unquote_plus(password),
    db=database,
    charset='utf8mb4'
)
cursor = conn.cursor()

# -------------------------- 1. CSV 병합 --------------------------
input_dir = "data_folder"
csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
all_data = []

for file_path in csv_files:
    print(f"📂 불러오는 중: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='cp949')
        all_data.append(df)
    except Exception as e:
        print(f"❌ 오류: {file_path}, 이유: {e}")

if not all_data:
    print("❌ 처리할 데이터가 없습니다.")
    exit()

merged_df = pd.concat(all_data, ignore_index=True)

# -------------------------- 2. '상품' 필터링 --------------------------
if 'BULK_GRAD_NM' in merged_df.columns:
    merged_df = merged_df[merged_df['BULK_GRAD_NM'] == '상품']

# -------------------------- 3. 날짜 정제 및 필수 컬럼 체크 --------------------------
required_columns = ['CTNP_NM', 'MRKT_NM', 'PDLT_NM', 'PRCE_REG_YMD']
if not all(col in merged_df.columns for col in required_columns):
    print("❌ 필수 컬럼이 누락되어 있습니다.")
    exit()

# 날짜 컬럼: 문자열 공백 제거 후 변환, 유효하지 않은 날짜 제거
merged_df['PRCE_REG_YMD'] = merged_df['PRCE_REG_YMD'].astype(str).str.strip()
merged_df['PRCE_REG_YMD'] = pd.to_datetime(merged_df['PRCE_REG_YMD'], errors='coerce')
merged_df = merged_df.dropna(subset=['PRCE_REG_YMD'])
merged_df['PRCE_REG_YMD'] = merged_df['PRCE_REG_YMD'].dt.date

# -------------------------- 중간 디버깅 출력 --------------------------
print("✅ 병합된 데이터 크기:", merged_df.shape)
print("📅 날짜 샘플:", merged_df['PRCE_REG_YMD'].dropna().unique()[:5])
print("🏢 지역 조합 수:", merged_df[['CTNP_NM', 'MRKT_NM']].drop_duplicates().shape)
print("📦 품목 예시:", merged_df['PDLT_NM'].unique()[:5])

# -------------------------- 4. 그룹 정렬 --------------------------
grouped = merged_df.groupby(['CTNP_NM', 'MRKT_NM', 'PDLT_NM'])
all_groups = []

for (ctnp, mrkt, pdlt), group in grouped:
    group_sorted = group.sort_values(by='PRCE_REG_YMD')
    group_sorted = group_sorted.dropna(axis=1, how='all')
    group_sorted['CTNP_NM'] = ctnp
    group_sorted['MRKT_NM'] = mrkt
    group_sorted['PDLT_NM'] = pdlt
    all_groups.append(group_sorted)

final_df = pd.concat(all_groups, ignore_index=True)

# -------------------------- 5. CREATE TABLE 자동 생성 --------------------------
def get_column_type(col, dtype):
    if col in ['CTNP_NM', 'MRKT_NM', 'PDLT_NM']:  # PK 대상은 길이 줄임
        return 'VARCHAR(100)'
    elif dtype == 'object':
        return 'VARCHAR(255)'
    elif dtype == 'float64':
        return 'FLOAT'
    elif dtype == 'int64':
        return 'INT'
    elif dtype == 'datetime64[ns]':
        return 'DATE'
    elif dtype == 'bool':
        return 'BOOLEAN'
    else:
        return 'VARCHAR(255)'

def generate_create_table_sql(df: pd.DataFrame, table_name: str, primary_keys: list[str]):
    column_defs = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sql_type = get_column_type(col, dtype)
        column_defs.append(f"`{col}` {sql_type}")

    pk_sql = f", PRIMARY KEY ({', '.join([f'`{k}`' for k in primary_keys])})"
    return f"CREATE TABLE IF NOT EXISTS `{table_name}` (\n  " + ",\n  ".join(column_defs) + pk_sql + "\n) CHARACTER SET utf8mb4;"

primary_keys = ['CTNP_NM', 'MRKT_NM', 'PDLT_NM', 'PRCE_REG_YMD']
create_table_sql = generate_create_table_sql(final_df, table_name, primary_keys)

try:
    cursor.execute(create_table_sql)
    conn.commit()
    print("✅ 테이블 자동 생성 완료")
except Exception as e:
    print(f"❌ 테이블 생성 실패: {e}")
    conn.rollback()

# -------------------------- 6. INSERT IGNORE 수행 --------------------------
cols = final_df.columns.tolist()
col_placeholders = ", ".join(["%s"] * len(cols))
col_names = ", ".join([f"`{col}`" for col in cols])
insert_sql = f"INSERT IGNORE INTO `{table_name}` ({col_names}) VALUES ({col_placeholders})"

# NaN → None 변환
rows = [
    tuple(None if pd.isna(val) else val for val in row)
    for row in final_df.itertuples(index=False)
]

try:
    cursor.executemany(insert_sql, rows)
    conn.commit()
    print(f"✅ 중복 제외하고 {cursor.rowcount}개의 레코드가 삽입되었습니다.")
except Exception as e:
    print(f"❌ INSERT 실패: {e}")
    conn.rollback()
finally:
    cursor.close()
    conn.close()