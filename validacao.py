import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# Carregar os dados de teste
url = "https://www.football-data.co.uk/mmz4281/2324/D1.csv" # Coloque aqui sua URL 
data = pd.read_csv(url)
colunas = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
data = data[colunas]
X_test = data.drop(['FTR'], axis=1)
y_test = data['FTR']

# Definir as transformações para colunas categóricas e numéricas
categorical_features = ['HomeTeam', 'AwayTeam']
numeric_features = ['FTHG', 'FTAG', 'HTHG', 'HTAG']

# Criar o pipeline de transformação
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

# Aplicar o pré-processamento ao conjunto de dados de teste
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_test_preprocessed = pipeline.fit_transform(X_test)
X_test_preprocessed_dense = X_test_preprocessed.toarray()

# Definir o modelo da mesma forma que durante o treinamento
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Carregar o modelo do arquivo de checkpoint
checkpoint = torch.load('checkpoint.pth')
input_size = X_test_preprocessed_dense.shape[1]
hidden_size = 100
output_size = 3  # 3 classes (Away, Draw, Home)
model = Net(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint['model_state_dict'])

# Avaliar o modelo carregado
model.eval()
with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test_preprocessed_dense))
_, predicted = torch.max(y_pred, 1)
accuracy = accuracy_score(y_test.map({'A': 0, 'D': 1, 'H': 2}).values, predicted)
conf_matrix = confusion_matrix(y_test.map({'A': 0, 'D': 1, 'H': 2}).values, predicted)
print("Acurácia do modelo:", accuracy)

# Plotar a matriz de confusão
labels = ['A','D','H']

plt.figure(figsize=(8,6))
plt.bar(labels, conf_matrix[0], color='r', label='Actual Away')
plt.bar(labels, conf_matrix[1], bottom=conf_matrix[0], color='g', label='Actual Draw')
plt.bar(labels, conf_matrix[2], bottom=conf_matrix[0]+conf_matrix[1], color='b', label='Actual Home')
plt.xlabel('Predicted Label')
plt.ylabel('Counts')
plt.title('Matrix confusão')
plt.legend()
plt.show()
