import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

# Carregar os dados
url = "https://www.football-data.co.uk/mmz4281/2324/D1.csv"
data = pd.read_csv(url)

# Selecionar as colunas relevantes
colunas = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
data = data[colunas]

# Dividir as colunas em características e rótulos
X = data.drop(['FTR'], axis=1)
y = data['FTR']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir as transformações para colunas categóricas e numéricas
categorical_features = ['HomeTeam', 'AwayTeam']
numeric_features = ['FTHG', 'FTAG', 'HTHG', 'HTAG']

# Criar os pipelines de transformação
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Aplicar as transformações
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Criar o pipeline completo com pré-processamento
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Aplicar o pré-processamento ao conjunto de dados de treinamento
X_train_preprocessed = pipeline.fit_transform(X_train)

# Convertendo a matriz esparsa em uma matriz densa
X_train_preprocessed_dense = X_train_preprocessed.toarray()

# Calcular o número de entradas (características)
input_size = X_train_preprocessed_dense.shape[1]
print("Número de entradas (características):", input_size)

# Treinar o modelo usando PyTorch
hidden_size = 100
output_size = 3  # 3 classes (Away, Draw, Home)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

start = time.perf_counter()
epochs = 1000  # número de épocas de treinamento
checkpoint_interval = 1000  # intervalo para salvar o checkpoint
errors = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # Fazer o forward
    y_pred = model(torch.FloatTensor(X_train_preprocessed_dense))
    # Cálculo do erro
    loss = criterion(y_pred, torch.LongTensor(y_train.map({'A': 0, 'D': 1, 'H': 2}).values))
    errors.append(loss.item())
    if epoch % 100 == 0:
        print(f'Época: {epoch} Loss: {loss.item()}')
    # Backpropagation
    loss.backward()
    optimizer.step()
    # Salvar o checkpoint
    if epoch % checkpoint_interval == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, 'Bundesligacheckpoint.pth')
# Testar o modelo já treinado
end = time.perf_counter()
tempo_total = end-start

# Aplicar o pré-processamento ao conjunto de dados de teste
X_test_preprocessed = pipeline.transform(X_test)

# Convertendo a matriz esparsa em uma matriz densa
X_test_preprocessed_dense = X_test_preprocessed.toarray()

# Avaliar o modelo treinado
model.eval()
y_pred = model(torch.FloatTensor(X_test_preprocessed_dense))
_, predicted = torch.max(y_pred, 1)

# Mapear os valores preditos para 'A', 'D', 'H'
predicted_labels = np.vectorize({0: 'A', 1: 'D', 2: 'H'}.get)(predicted.numpy())

# Calcular a acurácia
accuracy = accuracy_score(y_test, predicted_labels)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, predicted_labels)

# Exibir a acurácia e a matriz de confusão
print("Acurácia do modelo:", accuracy)
print("Matriz de confusão:")
print(conf_matrix)

# Remover a coluna 'HomeTeam' antes de calcular a média
df_media_times = data.drop(['HomeTeam'], axis=1).groupby(['AwayTeam']).mean().reset_index()

dict_media_times = df_media_times.set_index('AwayTeam').T.to_dict('list')

# Coletar os dados das partidas futuras
dados_partidas_futuras = {
    'HomeTeam': ['TimeCasa1', 'TimeCasa2'],  # Nome dos times da casa
    'AwayTeam': ['TimeFora1', 'TimeFora2'],  # Nome dos times visitantes
    'FTHG': [1, 0],  # Gols marcados em casa na primeira etapa
    'FTAG': [0, 2],  # Gols marcados fora na primeira etapa
    'HTHG': [0, 1],  # Gols marcados em casa na segunda etapa
    'HTAG': [1, 1]   # Gols marcados fora na segunda etapa
}

# Criar DataFrame com os dados das partidas futuras
df_partidas_futuras = pd.DataFrame(dados_partidas_futuras)

# Calcular a média das estatísticas dos times da casa para as partidas futuras
df_partidas_futuras['FTHG_mean'] = df_partidas_futuras['HomeTeam'].map(lambda x: dict_media_times[x][0])
df_partidas_futuras['HTHG_mean'] = df_partidas_futuras['HomeTeam'].map(lambda x: dict_media_times[x][1])

# Calcular a média das estatísticas dos times visitantes para as partidas futuras
df_partidas_futuras['FTAG_mean'] = df_partidas_futuras['AwayTeam'].map(lambda x: dict_media_times[x][0])
df_partidas_futuras['HTAG_mean'] = df_partidas_futuras['AwayTeam'].map(lambda x: dict_media_times[x][1])

# Exibir o DataFrame com as estatísticas médias das partidas futuras
print("\nResultados previstos das partidas futuras:")
print(df_partidas_futuras)
