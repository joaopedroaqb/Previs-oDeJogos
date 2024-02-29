import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn

# Definir o tamanho correto das características de entrada (input_size) usado durante o treinamento original
input_size = 40  # Verifique o valor correto com base no treinamento original do modelo

# Carregar o modelo treinado
checkpoint = torch.load('Bundesligacheckpoint.pth')

# Obter as dimensões das camadas do modelo carregado
hidden_size = checkpoint['model_state_dict']['fc1.weight'].size(0)  # Corrigido para obter o primeiro valor da tupla de forma
output_size = checkpoint['model_state_dict']['fc2.weight'].size(0)

# Definir o modelo de rede neural
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Criar uma instância do modelo
model = Net(input_size, hidden_size, output_size)

# Carregar o estado do modelo a partir do checkpoint
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Definir strict como False para ignorar incompatibilidades de forma

# Avaliar o modelo treinado
model.eval()

# Coletar dados das partidas futuras
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

# Selecionar apenas as características relevantes para fazer as previsões (igual ao conjunto de treinamento)
df_medias_futuras = df_partidas_futuras[['FTHG', 'FTAG', 'HTHG', 'HTAG']]

# Definir as transformações para colunas numéricas
numeric_features = ['FTHG', 'FTAG', 'HTHG', 'HTAG']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Aplicar as transformações
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Aplicar o mesmo pré-processamento que foi feito nos dados de treinamento
X_partidas_futuras_preprocessed = preprocessor.fit_transform(df_medias_futuras)

# Fazer previsões sobre os resultados das partidas futuras
y_pred_partidas_futuras = model(torch.FloatTensor(X_partidas_futuras_preprocessed))

# Mapear os resultados previstos de volta para as classes originais ('A', 'D', 'H')
_, predicted_partidas_futuras = torch.max(y_pred_partidas_futuras, 1)
resultados_previstos = predicted_partidas_futuras.map({0: 'A', 1: 'D', 2: 'H'})

# Exibir os resultados previstos
print("Resultados previstos das partidas futuras:")
for i, (home, away, result) in enumerate(zip(df_partidas_futuras['HomeTeam'], df_partidas_futuras['AwayTeam'], resultados_previstos)):
    print(f"Jogo {i+1}: {home} vs {away} - Resultado previsto: {result}")
