# LSTM Stock Price Prediction API

Este projeto implementa uma API para previsão de preços de ações usando um modelo LSTM (Long Short-Term Memory). A API recebe dados históricos de preços de ações e retorna previsões para os próximos dias. Além disso, a API inclui um painel de monitoramento para rastrear o desempenho em tempo real.

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Instalação](#instalação)
3. [Uso da API](#uso-da-api)
4. [Documentação da API](#documentação-da-api)
5. [Monitoramento](#monitoramento)
6. [Exemplos](#exemplos)
7. [Contribuição](#contribuição)
8. [Licença](#licença)

---


## Visão Geral

O projeto consiste em:

- Um modelo LSTM treinado para prever preços futuros de ações com base em dados históricos.
- Uma API RESTful construída com Flask para servir o modelo.
- Um painel de monitoramento usando **Flask-MonitoringDashboard** para rastrear métricas como tempo de resposta e utilização de recursos.

---

## Instalação

Siga os passos abaixo para configurar o projeto localmente.

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)

### Passos

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/lstm-stock-prediction-api.git
   cd lstm-stock-prediction-api
   ```
   
2.  Crie um ambiente virtual (opcional, mas recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
   ```
3.  Instale as dependências:

   ```bash
    pip install -r requirements.txt
   ```
4.  Baixe o modelo pré-treinado (se necessário):

    ``Coloque o arquivo lstm_stock_model.pth na raiz do projeto``
    
5.  Execute a API:

    ```bash
    python app.py
    ```
A API estará disponível em http://localhost:5000.

---

## Docker

```` bash
docker-compose build
docker-compose up
````
A API estará disponível em http://localhost:5000.

---
## Uso da API
### Endpoint Principal
- POST /predict: Recebe dados históricos e retorna previsões de preços futuros.

Exemplo de Requisição

        curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{
          "data": [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.1, 2.1, 3.1, 4.1, 5.1],
            [1.2, 2.2, 3.2, 4.2, 5.2]
          ]
        }

Exemplo de Resposta
```json
    {
      "prediction": [100.5, 101.2, 102.0]
    }
```
---
## Documentação da API

A documentação interativa da API está disponível em:

```bash
http://localhost:5000/api/docs
```

Nela, você pode:

- Ver todos os endpoints disponíveis.

- Testar os endpoints diretamente na interface.

- Ver exemplos de requisições e respostas.
---
## Monitoramento
O painel de monitoramento está disponível em:

```bash
http://localhost:5000/dashboard
```
Nele, você pode visualizar métricas como:

- Tempo de resposta das requisições.

- Utilização de CPU e memória.

- Número de requisições por endpoint.
---
## Exemplos
### Usando Python

Aqui está um exemplo de como usar a API com Python:
````python
    import requests
    
    url = "http://localhost:5000/predict"
    data = {
        "data": [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.1, 2.1, 3.1, 4.1, 5.1],
            [1.2, 2.2, 3.2, 4.2, 5.2]
        ]
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("Previsão:", response.json())
    else:
        print("Erro:", response.status_code, response.text)
````
---
## Contribuição
Contribuições são bem-vindas! Siga os passos abaixo:

1. Faça um fork do projeto.

2. Crie uma branch para sua feature (``git checkout -b feature/nova-feature``).

3. Commit suas mudanças (``git commit -m 'Adiciona nova feature'``).

4. Push para a branch (``git push origin feature/nova-feature``).

5. Abra um Pull Request.
---
## Licença
Esse `README.md` cobre os principais aspectos de um projeto Flask, incluindo a descrição do projeto, instalação, estrutura dos arquivos, uso e como contribuir. Ajuste conforme necessário para refletir os detalhes específicos do seu projeto.

---
## Contato

Para dúvidas ou sugestões, entre em contato:

- Nome: Seu Nome

- Email: [seu-email@example.com](mailto:seu-email@example.com)

- GitHub: [Seu Usuario](https://github.com/seu-usuario)
---
## Agradecimentos

- À comunidade de código aberto por fornecer ferramentas incríveis.
- Aos tutoriais e documentações que ajudaram no desenvolvimento deste projeto.
