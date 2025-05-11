import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정 (운영체제에 따라)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux (예: Colab, Ubuntu)
    import matplotlib
    matplotlib.font_manager._rebuild()
    fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


# ------------------------ #
# 1. 데이터 불러오기 및 전처리
# ------------------------ #
input_path = "./split_files/서울_A-유통_감자.csv"
input_filename = os.path.basename(input_path)
base_name = os.path.splitext(input_filename)[0]
df = pd.read_csv(input_path, encoding='euc-kr')
df = df[["PRCE_REG_YMD", "PDLT_PRCE"]].dropna()
df["PRCE_REG_YMD"] = pd.to_datetime(df["PRCE_REG_YMD"], format="%Y%m%d")
df["PDLT_PRCE"] = pd.to_numeric(df["PDLT_PRCE"], errors="coerce")
df = df.dropna().sort_values("PRCE_REG_YMD")

# ------------------------ #
# 2. EDA - 기본 통계 및 시각화
# ------------------------ #
print("데이터 개수:", len(df))
print("날짜 범위:", df["PRCE_REG_YMD"].min(), "→", df["PRCE_REG_YMD"].max())
print("가격 통계:\n", df["PDLT_PRCE"].describe())

plt.figure(figsize=(12, 5))
plt.plot(df["PRCE_REG_YMD"], df["PDLT_PRCE"], marker='o')
plt.title("감자 가격 추이")
plt.xlabel("날짜"); plt.ylabel("가격"); plt.grid(True)
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df["PDLT_PRCE"], bins=20, kde=True)
plt.title("가격 분포")
plt.tight_layout(); plt.show()

df["price_pct"] = df["PDLT_PRCE"].pct_change() * 100
plt.figure(figsize=(12, 4))
plt.plot(df["PRCE_REG_YMD"], df["price_pct"], color='orange')
plt.axhline(0, color='gray', linestyle='--')
plt.title("일일 가격 증감률 (%)"); plt.grid(True)
plt.tight_layout(); plt.show()

df["MA_7"] = df["PDLT_PRCE"].rolling(7).mean()
df["MA_14"] = df["PDLT_PRCE"].rolling(14).mean()
plt.figure(figsize=(12, 5))
plt.plot(df["PRCE_REG_YMD"], df["PDLT_PRCE"], label="실제 가격", alpha=0.5)
plt.plot(df["PRCE_REG_YMD"], df["MA_7"], label="7일 이동 평균", linestyle='--')
plt.plot(df["PRCE_REG_YMD"], df["MA_14"], label="14일 이동 평균", linestyle='-.')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

df["weekday"] = df["PRCE_REG_YMD"].dt.day_name()
weekday_avg = df.groupby("weekday")["PDLT_PRCE"].mean().reindex([
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
])
plt.figure(figsize=(8, 4))
weekday_avg.plot(kind='bar', color='skyblue')
plt.title("요일별 평균 가격")
plt.ylabel("평균 가격")
plt.grid(axis='y'); plt.tight_layout(); plt.show()

# ------------------------ #
# 2-2. EDA 결과 요약 저장 (CSV + SQLite)
# ------------------------ #

# 1. 작물 이름 추출 (예: '서울_A-유통_감자' → '감자')
crop_name = base_name.split("_")[-1]  # 파일명 기준으로 마지막 '_' 뒤가 작물명

# 2. EDA 데이터 구성
eda_df = df[["PRCE_REG_YMD", "PDLT_PRCE", "price_pct", "MA_7", "MA_14"]].copy()
eda_df["weekday"] = df["PRCE_REG_YMD"].dt.day_name()
eda_df["crop_name"] = crop_name

# 3. CSV 저장
eda_output_filename = f"{base_name}_EDA_요약.csv"
eda_df.to_csv(eda_output_filename, index=False, encoding='utf-8-sig')
print(f"📊 EDA 요약 파일이 '{eda_output_filename}'로 저장되었습니다.")

# 4. SQLite 파일로 저장
import sqlite3

sqlite_filename = f"{base_name}_EDA.sqlite"
table_name = "eda_result"

# SQLite 연결 및 저장
conn = sqlite3.connect(sqlite_filename)
eda_df.to_sql(table_name, conn, if_exists='replace', index=False)
conn.close()

print(f"🗂️ SQLite 파일 '{sqlite_filename}'에 테이블 '{table_name}'로 저장 완료!")
print("📌 이제 이 파일을 MySQL에서 불러오면 됩니다.")

# ------------------------ #
# 3. 날짜 보간 및 시계열 생성
# ------------------------ #
full_dates = pd.date_range(start=df["PRCE_REG_YMD"].min(), end=df["PRCE_REG_YMD"].max())
df_full = pd.DataFrame({"PRCE_REG_YMD": full_dates})
df_merged = pd.merge(df_full, df[["PRCE_REG_YMD", "PDLT_PRCE"]], on="PRCE_REG_YMD", how="left")

# ffill로 초기 보간 후 학습용 모델 구축
df_merged["PDLT_PRCE_ffill"] = df_merged["PDLT_PRCE"].fillna(method='ffill')
scaler = MinMaxScaler()
price_scaled = scaler.fit_transform(df_merged[["PDLT_PRCE_ffill"]])

# 시퀀스 생성 함수
def create_sequences(data, input_window=30, output_window=14):
    X, y = [], []
    for i in range(len(data) - input_window - output_window):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window:i+input_window+output_window])
    return np.array(X), np.array(y)

# 초기 모델로 누락 데이터 예측
X_temp, y_temp = create_sequences(price_scaled, 30, 14)
temp_dataset = TensorDataset(torch.tensor(X_temp, dtype=torch.float32), torch.tensor(y_temp, dtype=torch.float32))
temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=True)

class CNNBiLSTM(nn.Module):
    def __init__(self, input_size=1, cnn_channels=32, lstm_hidden=128, output_len=14):
        super(CNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.bilstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, output_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        output, _ = self.bilstm(x)
        out = self.fc(output[:, -1, :])
        return out.unsqueeze(-1)

model = CNNBiLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    for xb, yb in temp_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 누락 구간 예측 보간
filled_prices = df_merged["PDLT_PRCE"].values.copy()
for i in range(len(df_merged)):
    if pd.isna(filled_prices[i]) and i >= 30:
        seq = price_scaled[i-30:i]
        input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor).squeeze().numpy()[0]
        filled_prices[i] = scaler.inverse_transform([[pred]])[0, 0]

df_merged["PDLT_PRCE_filled"] = filled_prices

# ------------------------ #
# 4. 최종 모델 학습 및 예측
# ------------------------ #
df_model = df_merged.dropna(subset=["PDLT_PRCE_filled"])
price_scaled_final = scaler.fit_transform(df_model[["PDLT_PRCE_filled"]])
X_final, y_final = create_sequences(price_scaled_final, 30, 14)
final_dataset = TensorDataset(torch.tensor(X_final, dtype=torch.float32), torch.tensor(y_final, dtype=torch.float32))
final_loader = DataLoader(final_dataset, batch_size=16, shuffle=True)

model = CNNBiLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in final_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/100, Loss: {total_loss:.4f}")

model.eval()
with torch.no_grad():
    recent_seq = torch.tensor(price_scaled_final[-30:], dtype=torch.float32).unsqueeze(0)
    pred_scaled = model(recent_seq).squeeze().numpy()
    pred_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

future_dates = pd.date_range(df_model["PRCE_REG_YMD"].max() + pd.Timedelta(days=1), periods=14)
plt.figure(figsize=(10, 5))
plt.plot(df_model["PRCE_REG_YMD"].values[-60:], df_model["PDLT_PRCE_filled"].values[-60:], label="실제 가격 (최근 60일)")
plt.plot(future_dates, pred_price, marker='o', label="예측 가격 (향후 14일)")
plt.title("감자 가격 예측 (CNN-BiLSTM)")
plt.xlabel("날짜"); plt.ylabel("가격")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

output_filename = f"{base_name}_예측_14일.csv"
result_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Price": pred_price.astype(int)
})
result_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"예측 결과가 '{output_filename}'에 저장되었습니다.")
