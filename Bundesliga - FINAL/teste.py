import pandas as pd
import numpy as np
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# Verificar e instalar o módulo openpyxl se necessário
try:
    import openpyxl
except ModuleNotFoundError:
    print("Instalando o módulo openpyxl...")
    os.system('pip install openpyxl')
    import openpyxl

# Definição da rede neural
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.Tanh = torch.nn.Sigmoid() 

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.Tanh(output)
        return output

# Função para prever os resultados de uma rodada futura
def preverRodada(nome, rodadaInput):
    columns_data = ['HomeTeam', 'AwayTeam', 'HTHG', 'HTAG']
    data = pd.read_csv('https://www.football-data.co.uk/mmz4281/2324/D1.csv', header=0, sep=',', usecols=columns_data)
    data = data.replace(np.nan, 0)
    df_rodada = pd.DataFrame(rodadaInput)
    df_rodada['HTHG'] = 0
    df_rodada['HTAG'] = 0
    df_mediasHome = data.groupby('HomeTeam')['HTHG'].mean()
    df_mediasAway = data.groupby('AwayTeam')['HTAG'].mean()
    for index, rodada in df_rodada.iterrows():
        for itemRodada in rodada:
            for timeHome in df_mediasHome.index:
                if (itemRodada == timeHome):
                    df_rodada.loc[index, 'HTHG'] = df_mediasHome[timeHome]+(df_mediasAway[timeHome]*0.3)
    for index, rodada in df_rodada.iterrows():
        for itemRodada in rodada:
            for timeAway in df_mediasAway.index:
                if (itemRodada == timeAway):
                    df_rodada.loc[index, 'HTAG'] = df_mediasAway[timeAway]+(df_mediasHome[timeAway]*0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_rodada = df_rodada.drop(labels=["HomeTeam", "AwayTeam"], axis=1)
    print(df_rodada)
    test_input = torch.FloatTensor((df_rodada.values)/10)
    input_size = test_input.size()[1]
    hidden_size = 4
    model = Net(input_size, hidden_size)
    try:
        model.load_state_dict(torch.load(f'{nome}.pth'))
    except FileNotFoundError:
        print(f"O arquivo '{nome}.pth' não foi encontrado. Treinando o modelo...")
        # Carregar os dados de treinamento
        url = "https://www.football-data.co.uk/mmz4281/2324/D1.csv"
        data = pd.read_csv(url)
        # Selecionar as colunas relevantes
        colunas = ['HomeTeam', 'AwayTeam', 'HTHG', 'HTAG', 'FTHG', 'FTAG', 'FTR']
        data = data[colunas]
        # Dividir os dados em características e rótulos
        X = data[['HTHG', 'HTAG']] / 10
        y = np.where(data['FTHG'] + data['FTAG'] > 1.5, 1, 0)
        # Dividir os dados em conjuntos de treinamento e teste
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        # Normalizar os dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        # Treinar o modelo
        input_size = X_train.shape[1]
        hidden_size = 4
        model = Net(input_size, hidden_size)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # Treinamento
        model.train()
        epochs = 1000
        errors = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Fazer o forward
            y_pred = model(torch.FloatTensor(X_train))
            # Cálculo do erro
            loss = criterion(y_pred.squeeze(), torch.FloatTensor(y_train))
            errors.append(loss.item())
            if epoch % 100 == 0:
                print(f'Época: {epoch} Loss: {loss.item()}')
            # Backpropagation
            loss.backward()
            optimizer.step()
        # Salvar o modelo treinado
        torch.save(model.state_dict(), f'{nome}.pth')
        print(f"Modelo treinado e salvo como '{nome}.pth'.")
    model.eval()
    y_pred = model(test_input)
    y_pred = y_pred.detach().numpy()
    y_pred = y_pred.tolist()
    for i in range(0, len(y_pred)):
        y_pred[i][0] = round(y_pred[i][0], 0)
    df_rodada = pd.DataFrame(rodadaInput)
    df_resultados = pd.DataFrame({'Time Casa': df_rodada["HomeTeam"], 'Previsão': y_pred, 'Time Fora': df_rodada["AwayTeam"]})
    df_resultados.to_excel('tabela_resultados.xlsx', index=False)
    return y_pred


# Prever resultados para uma rodada futura
rodada_input = {
    'HomeTeam': ['TimeCasa1', 'TimeCasa2'],  # Nome dos times da casa
    'AwayTeam': ['TimeFora1', 'TimeFora2'],  # Nome dos times visitantes
    'HTHG': [0, 0],  # Média de gols marcados em casa na primeira etapa
    'HTAG': [0, 0]   # Média de gols marcados fora na primeira etapa
}

preverRodada("modelo_futebol", pd.DataFrame(rodada_input))
