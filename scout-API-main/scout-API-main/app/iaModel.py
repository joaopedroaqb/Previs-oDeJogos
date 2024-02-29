import pandas as pd
import numpy as np
import torch
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.Tanh = torch.nn.Tanh() 
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.Tanh(output)
        return output

def redeNeural(nome, momentum, lr, epocas, hiddenSize, datasetNome, entradas):
    entradas = entradas.replace("[", "")
    entradas = entradas.replace("]", "")
    entradas = entradas.split(",")
    momentum = float(momentum)
    lr = float(lr)
    epocas = int(epocas)
    hiddenSize = int(hiddenSize)
    start = time.perf_counter()
    datasetNome = datasetNome+'.csv'
    columns_data = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
    for entrada in entradas:
        columns_data.append(entrada)
    data = pd.read_csv('app/data/dataset/'+datasetNome, header=0, sep=',', usecols=columns_data)
    data = data.replace(np.nan, 0)
    training = data.iloc[:-10]
    test = data.iloc[-10:]

    training = training.sample(frac=1)
    test = test.sample(frac=1)

    nomes = data[['HomeTeam', 'AwayTeam']]
    data1 = data.iloc[:, 5:]
    data2 = data[['FTR']]
    data_transformed = data2.replace({"H":1,"D":0,"A":-1})
    input = data1.replace({"H":1,"D":0,"A":-1})
    output = data_transformed[['FTR']]
    nomes_training = nomes[:-10]
    nomes_test = nomes[-10:]
    #Normalização de dados
    for e in range(len(input.columns)): 
        max = input.iloc[:, e].max() #checar o valor maximo de cada coluna
        if max < 10:
            input.iloc[:, e] /= 10
            
        elif max < 100:
            input.iloc[:, e] /= 100
        else:
            print("Error in normalization! Please check!")

    training_input = input[:-10]
    training_output= output[:-10]
    test_input = input[-10:]
    test_output = output[-10:]
    # Convertendo para tensor
    training_input = torch.FloatTensor(training_input.values)
    training_output = torch.FloatTensor(training_output.values)
    test_input = torch.FloatTensor(test_input.values)
    test_output = torch.FloatTensor(test_output.values)

    # Criar a instância do modelo
    input_size = training_input.size()[1] 
    hidden_size = hiddenSize
    model = Net(input_size, hidden_size) 
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum)

    # Treinamento
    model.train()
    epochs = epocas
    errors = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Fazer o forward
        y_pred = model(training_input)
        # Cálculo do erro
        loss = criterion(y_pred.squeeze(), training_output.squeeze())
        errors.append(loss.item())
        if epoch % 10000 == 0:
            print(f'Época: {epoch} Loss: {loss.item()}')
        # Backpropagation
        loss.backward()
        optimizer.step()
    # Testar o modelo já treinado
    end = time.perf_counter()
    tempo_total = end-start
    model.eval()
    y_pred = model(test_input)
    erro_pos_treinamento = criterion(y_pred.squeeze(), test_output.squeeze())
    predicted = y_pred.detach().numpy()
    real = test_output.numpy()
    torch.save(model.state_dict(), "app/data/modeloTreinado.pth")
    erro_pos_treinamento = erro_pos_treinamento.item()/len(test_output)
    return tempo_total, erro_pos_treinamento, predicted, real

# Tratamento String nome arquivo modelo
def tratarString(nome):
    nome = nome.lower()
    nome = nome.replace(' ', '-')

def preverRodada(datasetNome, entradas, rodadaInput):
    entradas = entradas.replace("[", "")
    entradas = entradas.replace("]", "")
    entradas = entradas.split(",")
    columns_data = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', "HTR"]
    datasetNome = datasetNome+'.csv'
    for entrada in entradas:
        columns_data.append(entrada)
    data = pd.read_csv('app/data/dataset/'+datasetNome, header=0, sep=',', usecols=columns_data)
    data = data.drop(labels=["FTHG", "FTAG", "FTR", "HTR"], axis=1)
    data = data.replace(np.nan, 0)
    home_columns = [col for col in data.columns if 'H' in col]
    data_home = data.loc[:, home_columns]
    data_home = data_home.drop(["HTAG"], axis=1)
    df_mediasHome = data_home.groupby(['HomeTeam'], as_index=False).mean()
    away_columns = [col for col in data.columns if 'A' in col]
    data_away = data.loc[:, away_columns]
    df_mediasAway = data_away.groupby(['AwayTeam'], as_index=False).mean()
    df_rodada = pd.DataFrame(rodadaInput)
    for linha in range(0, len(df_rodada)):
        for coluna in entradas:
            df_rodada[coluna] = 0
    for linhaRodada in range(0, len(df_rodada)):
        for linhaAway in range(0, len(df_mediasAway)):
            for colunaAway in df_mediasAway:
                if (df_mediasAway.at[linhaAway, "AwayTeam"] == df_rodada.at[linhaRodada, "AwayTeam"]):
                    df_rodada.at[linhaRodada, colunaAway] = df_mediasAway.at[linhaAway, colunaAway]
    for linhaRodada in range(0, len(df_rodada)):
        for linhaHome in range(0, len(df_mediasHome)):
            for colunaAway in df_mediasHome:
                if (df_mediasHome.at[linhaHome, "HomeTeam"] == df_rodada.at[linhaRodada, "HomeTeam"]):
                    df_rodada.at[linhaRodada, colunaAway] = df_mediasHome.at[linhaHome, colunaAway]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_rodada = df_rodada.drop(labels=["HomeTeam", "AwayTeam"], axis=1)
    test_input = torch.FloatTensor(df_rodada.values)
    input_size = test_input.size()[1]
    hidden_size = 100
    model = Net(input_size, hidden_size)
    model.load_state_dict(torch.load('app/data/modeloTreinado.pth'))
    model.eval()
    y_pred = model(test_input)
    return y_pred

# Convertendo o modelo PyTorch para o formato sklearn
from sklearn.base import BaseEstimator, RegressorMixin

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, net):
        self.net = net

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(X).numpy()

# Carregar os dados
url = "https://www.football-data.co.uk/mmz4281/1920/D1.csv"
data = pd.read_csv(url)

# Selecionar as colunas relevantes
colunas = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
data = data[colunas]

# Dividir as colunas em características e rótulos
X = data.drop(['FTR'], axis=1)
y = data['FTR']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o pipeline de pré-processamento
# Você pode adicionar mais passos de pré-processamento conforme necessário
numeric_features = ['FTHG', 'FTAG', 'HTHG', 'HTAG']
categorical_features = ['HomeTeam', 'AwayTeam']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Aplicar o pré-processamento ao conjunto de dados de treinamento
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Convertendo a matriz esparsa em uma matriz densa
X_train_preprocessed_dense = X_train_preprocessed.toarray()

# Criando o modelo PyTorch
input_size = X_train_preprocessed_dense.shape[1]
hidden_size = 100
model = Net(input_size, hidden_size)

# Convertendo o modelo PyTorch para o formato sklearn
sklearn_model = PyTorchRegressor(model)

# Treinar o modelo
sklearn_model.fit(X_train_preprocessed_dense, y_train)

# Aplicar o pré-processamento ao conjunto de dados de teste
X_test_preprocessed = preprocessor.transform(X_test)

# Avaliar o modelo treinado
y_pred = sklearn_model.predict(X_test_preprocessed)
