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
import matplotlib.pyplot as plt

# Carregar os dados
url = "x" #Coloque aqui a URL desejada
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

# Treinar o modelo usando PyTorch
input_size = X_train_preprocessed_dense.shape[1]
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
epochs = 100000  # número de épocas de treinamento
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
accuracy = accuracy_score(y_test.map({'A': 0, 'D': 1, 'H': 2}).values, predicted)
conf_matrix = confusion_matrix(y_test.map({'A': 0, 'D': 1, 'H': 2}).values, predicted)
print("Acurácia do modelo:", accuracy)

# Plotar a matriz de confusão
labels = ['A', 'D', 'H']  # 'Away = Fora', 'Draw = Empate', 'Home = Casa'
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
plt.show()
