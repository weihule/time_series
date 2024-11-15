import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

def gen_data():
    # 生成虚拟气象数据集
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    temperature = np.sin(np.linspace(0, 2 * np.pi, len(dates))) * 10 + 20 + np.random.normal(0, 2, len(dates))
    humidity = np.cos(np.linspace(0, 2 * np.pi, len(dates))) * 20 + 60 + np.random.normal(0, 5, len(dates))
    wind_speed = np.abs(np.sin(np.linspace(0, 4 * np.pi, len(dates)))) * 5 + np.random.normal(0, 1, len(dates))

    # 构建 DataFrame
    df = pd.DataFrame({'Date': dates, 'Temperature': temperature, 'Humidity': humidity, 'WindSpeed': wind_speed})
    df.set_index('Date', inplace=True)

    # plt.figure(figsize=(12, 8))
    # plt.plot(df.index, df["Temperature"], label="Temperature")
    # plt.plot(df.index, df["Humidity"], label="Humidity")
    # plt.plot(df.index, df["WindSpeed"], label="WindSpeed")
    # plt.title("df draw")
    # plt.xlabel("date")
    # plt.ylabel("value")
    # plt.legend()
    # plt.show()

    return df

# 定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # x.shape [batch_size, seq_len, input_size]
        out, _ = self.lstm(x, (h0, c0))
        # print(f"---- h0.shape: {h0.shape} c0.shape: {c0.shape} out.shape: {out.shape} {out[:, -1, :].shape}")
        out = self.fc(out[:, -1, :])
        return out

# 转换为时间序列格式
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length][0]                    # 预测温度
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train(df):
    # 数据标准化
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # plt.figure(figsize=(12, 8))
    # plt.plot(df.index, data_scaled[:, 0], label="Temperature")
    # plt.plot(df.index, data_scaled[:, 1], label="Humidity")
    # plt.plot(df.index, data_scaled[:, 2], label="WindSpeed")
    # plt.title("df draw")
    # plt.xlabel("date")
    # plt.ylabel("value")
    # plt.legend()
    # plt.show()

    device = torch.device("cuda")

    seq_length = 30        # 使用过去30天的数据
    # X, y = create_sequences(data_scaled, seq_length)
    split_ratio = 0.8
    split_index = int(len(data_scaled) * split_ratio)
    X_train, y_train = create_sequences(data_scaled[:split_index], seq_length)
    X_test, y_test = create_sequences(data_scaled[split_index:], seq_length)
    print(f"data_scaled.shape: {data_scaled.shape} split_index: {split_index}\n"
          f"X_train.shape: {X_train.shape} y_train.shape: {y_train.shape}\n"
          f"X_test.shape: {X_test.shape} y_test.shape: {y_test.shape}")

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # 构建数据加载器
    batch_size = 16
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    # 初始化模型、损失函数和优化器
    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    output_size = 1
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # for batch_x, batch_y in train_loader:
    #     batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #     print(batch_x.shape, batch_y.shape)
    #     outputs = model(batch_x)
        # break

    # 训练模型
    epochs = 1000
    losses = []

    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            # print(outputs.shape, batch_y.shape)
            outputs = outputs.squeeze()
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color="red")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # 模型预测
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test, dtype=torch.float32).to(device)
                            ).cpu().numpy()

    # 可视化原始数据的趋势
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df['Temperature'], label="Temperature", color="blue")
    plt.plot(df.index, df['Humidity'], label="Humidity", color="green")
    plt.title("Original Weather Data Trends")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

    # 可视化预测结果和真实值对比
    plt.figure(figsize=(14, 8))
    print(split_index, seq_length, split_index-seq_length, len(df.index[split_index:]), len(y_test))
    plt.plot(df.index[split_index: len(df)-seq_length], y_test, label="True Temperature", color="blue")
    plt.plot(df.index[split_index: len(df)-seq_length], predictions, label="Predicted Temperature", color="orange")
    plt.title("Prediction vs True Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

    # 误差分布图
    errors = y_test - predictions.flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, color="purple", alpha=0.7)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()


def run():
    df = gen_data()
    train(df)

def test():
    criterion = nn.MSELoss()
    input = torch.randn(3, 1, requires_grad=True)
    target = torch.randn(3)
    loss = criterion(input, target)
    print(loss.item())


if __name__ == '__main__':
    run()
    # test()





