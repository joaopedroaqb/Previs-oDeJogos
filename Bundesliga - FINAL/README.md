# Previsão de Resultados de Futebol (Bundesliga)

Este projeto consiste em um código em Python para prever os resultados de uma rodada futura de partidas de futebol. Utiliza uma rede neural artificial (RNA) implementada com a biblioteca PyTorch para realizar as previsões.

## Requisitos

- Python 3.x
- Bibliotecas: pandas, numpy, torch, scikit-learn
- Módulo `openpyxl` (pode ser instalado usando `pip install openpyxl`)

## Funcionalidades Principais

### 1. Classe Net

Implementa a arquitetura da rede neural. A rede possui uma camada oculta com função de ativação ReLU e uma camada de saída com função de ativação Sigmoid.

### 2. Função `preverRodada`

- Recebe como entrada o nome do modelo e os dados da rodada a ser prevista.
- Carrega os dados históricos das partidas de futebol.
- Calcula a média de gols marcados em casa e fora para cada time.
- Normaliza os dados da rodada de entrada.
- Carrega o modelo treinado ou treina um novo se o modelo não existir.
- Realiza a previsão dos resultados para a rodada de entrada.
- Salva os resultados em um arquivo Excel.

## Uso

Para usar a função `preverRodada`, forneça o nome do modelo e os dados da rodada a ser prevista. Os dados da rodada devem incluir os nomes dos times da casa e visitantes, assim como a média de gols marcados em casa e fora na primeira etapa.

### Exemplo

```python
rodada_input = {
    'HomeTeam': ['TimeCasa1', 'TimeCasa2'],  # Nome dos times da casa
    'AwayTeam': ['TimeFora1', 'TimeFora2'],  # Nome dos times visitantes
    'HTHG': [0, 0],  # Média de gols marcados em casa na primeira etapa
    'HTAG': [0, 0]   # Média de gols marcados fora na primeira etapa
}

preverRodada("modelo_futebol", pd.DataFrame(rodada_input))
