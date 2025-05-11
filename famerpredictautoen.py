import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
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
crop_name = base_name.split("_")[-1]
eda_df = df[["PRCE_REG_YMD", "PDLT_PRCE", "price_pct", "MA_7", "MA_14"]].copy()
eda_df["weekday"] = df["PRCE_REG_YMD"].dt.day_name()
eda_df["crop_name"] = crop_name

eda_output_filename = f"{base_name}_EDA_요약.csv"
eda_df.to_csv(eda_output_filename, index=False, encoding='utf-8-sig')
print(f"📊 EDA 요약 파일이 '{eda_output_filename}'로 저장되었습니다.")

import sqlite3
sqlite_filename = f"{base_name}_EDA.sqlite"
table_name = "eda_result"
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

df_merged["PDLT_PRCE_ffill"] = df_merged["PDLT_PRCE"].ffill()  # 수정: fillna(method='ffill') → ffill()
scaler = MinMaxScaler()
price_scaled = scaler.fit_transform(df_merged[["PDLT_PRCE_ffill"]])

def create_sequences(data, input_window=30, output_window=14):
    X, y = [], []
    for i in range(len(data) - input_window - output_window):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window:i+input_window+output_window])
    return np.array(X), np.array(y)

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
# 3-2. 오토인코더 기반 이상치 탐지 및 처리
# ------------------------ #

# 오토인코더 정의
class Autoencoder(nn.Module):
    def __init__(self, input_size=30):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        if x.dim() == 3:  # (batch_size, 30, 1)인 경우
            x = x.squeeze(-1)  # (batch_size, 30)으로 변환
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 오토인코더 학습 데이터 준비
def create_ae_sequences(data, window=30):
    sequences = []
    for i in range(len(data) - window):
        sequences.append(data[i:i+window].flatten())  # (30, 1) -> (30,)
    return np.array(sequences)

# 가격 데이터를 정규화하여 오토인코더 학습
ae_scaler = MinMaxScaler()
price_scaled_ae = ae_scaler.fit_transform(df_merged[["PDLT_PRCE_filled"]])
ae_sequences = create_ae_sequences(price_scaled_ae, window=30)
print("ae_sequences shape:", ae_sequences.shape)  # 디버깅용

ae_dataset = TensorDataset(torch.tensor(ae_sequences, dtype=torch.float32))
ae_loader = DataLoader(ae_dataset, batch_size=16, shuffle=True)

# 오토인코더 학습
ae_model = Autoencoder(input_size=30)
ae_criterion = nn.MSELoss()
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)

for epoch in range(50):
    ae_model.train()
    total_loss = 0
    for batch in ae_loader:
        xb = batch[0]
        recon = ae_model(xb)
        loss = ae_criterion(recon, xb)
        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Autoencoder Epoch {epoch+1}/50, Loss: {total_loss:.4f}")

# 이상치 탐지
ae_model.eval()
recon_errors = []
with torch.no_grad():
    for seq in ae_sequences:
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        recon = ae_model(seq_tensor).squeeze().numpy()
        error = np.mean((seq - recon) ** 2)  # MSE
        recon_errors.append(error)

# 이상치 기준 설정 (상위 5% 오류를 이상치로 간주)
threshold = np.percentile(recon_errors, 95)
anomaly_indices = [i for i, error in enumerate(recon_errors) if error > threshold]

# 이상치 시각화
plt.figure(figsize=(12, 5))
plt.plot(df_merged["PRCE_REG_YMD"][30:], df_merged["PDLT_PRCE_filled"][30:], label="가격")
plt.scatter(df_merged["PRCE_REG_YMD"].iloc[anomaly_indices],
            df_merged["PDLT_PRCE_filled"].iloc[anomaly_indices],
            color='red', label="이상치", zorder=5)
plt.title("이상치 탐지 결과 (오토인코더)")
plt.xlabel("날짜"); plt.ylabel("가격")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# 이상치 교체 (사전 학습된 CNN-BiLSTM 사용)
corrected_prices = df_merged["PDLT_PRCE_filled"].values.copy()
for idx in anomaly_indices:
    if idx >= 30:
        seq = price_scaled_ae[idx-30:idx]
        input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor).squeeze().numpy()[0]  # 첫 번째 예측값 사용
        corrected_prices[idx] = ae_scaler.inverse_transform([[pred]])[0, 0]

df_merged["PDLT_PRCE_corrected"] = corrected_prices

# 이상치 교정 결과 시각화
plt.figure(figsize=(12, 5))
plt.plot(df_merged["PRCE_REG_YMD"], df_merged["PDLT_PRCE_filled"], label="원본 (보간 후)", alpha=0.5)
plt.plot(df_merged["PRCE_REG_YMD"], df_merged["PDLT_PRCE_corrected"], label="이상치 교정 후", linestyle='--')
plt.title("이상치 교정 전후 비교")
plt.xlabel("날짜"); plt.ylabel("가격")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ------------------------ #
# 4. 최종 모델 학습 및 예측
# ------------------------ #
df_model = df_merged.dropna(subset=["PDLT_PRCE_corrected"])  # 교정된 데이터 사용
price_scaled_final = scaler.fit_transform(df_model[["PDLT_PRCE_corrected"]])
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
plt.plot(df_model["PRCE_REG_YMD"].values[-60:], df_model["PDLT_PRCE_corrected"].values[-60:], label="실제 가격 (최근 60일)")
plt.plot(future_dates, pred_price, marker='o', label="예측 가격 (향후 14일)")
plt.title("감자 가격 예측 (CNN-BiLSTM, 이상치 교정 후)")
plt.xlabel("날짜"); plt.ylabel("가격")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

output_filename = f"{base_name}_예측_14일_이상치교정.csv"
result_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Price": pred_price.astype(int)
})
result_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"예측 결과가 '{output_filename}'에 저장되었습니다.")