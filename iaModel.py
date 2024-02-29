import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

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

def redeNeural(nome, momentum, lr, epocas, hiddenSize):
    momentum = float(momentum)
    lr = float(lr)
    epocas = int(epocas)
    hiddenSize = int(hiddenSize)
    start = time.perf_counter()
    columns_data = ['HomeTeam', 'AwayTeam', 'HTHG', 'HTAG']
    data = pd.read_csv('app/data/dataset/E0.csv', header=0, sep=',', usecols=columns_data)
    data = data.replace(np.nan, 0)
    sum_goals = data['HTAG']+data['HTHG']
    sum_goals = np.where(sum_goals > 1.5, 1, 0)
    data = data.assign(OVER2=sum_goals)
    data_input = data[['HTHG', 'HTAG']]/10
    data_output = data['OVER2']
    print(data_output)
    nomes = data[['HomeTeam', 'AwayTeam']]
    nomes_training = nomes[:-10]
    nomes_test = nomes[-10:]
    training_input = data_input[:-10]
    training_output= data_output[:-10]
    test_input = data_input[-10:]
    test_output = data_output[-10:]
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
        if epoch % 1000 == 0:
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
    real = test_output.detach().numpy()
    print(predicted)
    print(real)
    torch.save(model.state_dict(), f'app/data/modelos/{nome}.pth')
    erro_pos_treinamento = erro_pos_treinamento.item()/len(test_output)
    return tempo_total, erro_pos_treinamento, predicted, real

def preverRodada(nome, rodadaInput):

    columns_data = ['HomeTeam', 'AwayTeam', 'HTHG', 'HTAG']
    data = pd.read_csv('app/data/dataset/E0.csv', header=0, sep=',', usecols=columns_data)
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
    model.load_state_dict(torch.load(f'app/data/modelos/{nome}.pth'))
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