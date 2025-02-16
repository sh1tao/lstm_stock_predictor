# Usa uma imagem base do Python
FROM python:3.12-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia os arquivos de requisitos para o contêiner
COPY api_requirements.txt .

# Instala as dependências do projeto
RUN pip install --no-cache-dir -r api_requirements.txt

# Copia o código do projeto para o contêiner
COPY . .

# Expõe a porta 5000 (porta usada pela API)
EXPOSE 5000

# Comando para rodar a API quando o contêiner iniciar
CMD ["python", "app.py"]
