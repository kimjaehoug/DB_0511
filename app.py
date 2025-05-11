from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine

app = Flask(__name__)

# ===== MySQL 설정 =====
MYSQL_USER = 'famers'
MYSQL_PASSWORD = '1633'
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_DB = 'famers'
TABLE_NAME = 'prediction_result'

# ===== CNN-BiLSTM 모델 정의 =====
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

# ===== 시계열 시퀀스 생성 =====
def create_sequences(data, input_window=30, output_window=14):
    X, y = [], []
    for i in range(len(data) - input_window - output_window):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window:i+input_window+output_window])
    return np.array(X), np.array(y)

# ===== 예측 수행 함수 =====
def run_prediction(filepath, predict_days):
    df = pd.read_csv(filepath, encoding='euc-kr')
    df = df[["PRCE_REG_YMD", "PDLT_PRCE"]].dropna()
    df["PRCE_REG_YMD"] = pd.to_datetime(df["PRCE_REG_YMD"], format="%Y%m%d", errors="coerce")
    df["PDLT_PRCE"] = pd.to_numeric(df["PDLT_PRCE"], errors="coerce")
    df = df.dropna().sort_values("PRCE_REG_YMD")

    full_dates = pd.date_range(start=df["PRCE_REG_YMD"].min(), end=df["PRCE_REG_YMD"].max())
    df_full = pd.DataFrame({"PRCE_REG_YMD": full_dates})
    df_merged = pd.merge(df_full, df, on="PRCE_REG_YMD", how="left")
    df_merged["PDLT_PRCE"] = df_merged["PDLT_PRCE"].ffill()

    scaler = MinMaxScaler()
    price_scaled = scaler.fit_transform(df_merged[["PDLT_PRCE"]])

    X, y = create_sequences(price_scaled, input_window=30, output_window=predict_days)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = CNNBiLSTM(output_len=predict_days)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    recent_seq = torch.tensor(price_scaled[-30:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(recent_seq).squeeze().numpy()
    pred_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    last_date = df_merged["PRCE_REG_YMD"].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(predict_days)]

    result_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Price": pred_price.astype(float).round(2)
    })

    return result_df

# ===== 예측 API 라우트 =====
@app.route("/predict", methods=["POST"])
def predict_crop_price():
    try:
        data = request.get_json()
        region = data.get("region")
        subregion = data.get("subregion")
        crop = data.get("crop")
        predict_days = int(data.get("predict_days", 14))

        filename = f"{region}_{subregion}_{crop}.csv".replace(" ", "_")
        filepath = os.path.join("split_files", filename)

        if not os.path.exists(filepath):
            return jsonify({"error": f"❌ 파일 '{filename}'을 찾을 수 없습니다."}), 404

        pred_df = run_prediction(filepath, predict_days)

        # ===== MySQL 저장 =====
        from sqlalchemy import create_engine
        engine = create_engine(
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
        )

        pred_df["region"] = region
        pred_df["subregion"] = subregion
        pred_df["crop"] = crop

        pred_df.to_sql(TABLE_NAME, con=engine, if_exists="append", index=False)

        return jsonify({
            "message": "✅ 예측 성공",
            "predictions": pred_df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": f"❌ 예측 중 오류 발생: {str(e)}"}), 500

# ===== 서버 실행 =====
if __name__ == "__main__":
    app.run(debug=True)