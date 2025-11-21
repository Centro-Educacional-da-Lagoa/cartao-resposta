# 📝 Sistema Automatizado de Correção de Cartões Resposta

> **Sistema inteligente para correção automática de cartões resposta (gabaritos) usando OCR, visão computacional e IA**

## 🚀 Funcionalidades

- ✅ **Detecção automática** de gabaritos e folhas de resposta
- 🤖 **Extração de cabeçalho** com Google Gemini AI
- 📊 **Integração com Google Sheets** para armazenamento automático
- ☁️ **Sincronização com Google Drive** (download automático da pasta configurada)
- 🎯 **Alta precisão** na detecção de respostas marcadas
- 📁 **Processamento em lote** de múltiplos alunos
- 🔄 **Rate limiting** integrado para APIs
- 🐛 **Modo debug** com visualização detalhada
- 📱 **Suporte a PDF e imagens** (PNG, JPG, JPEG)

## 🎯 Como Funciona

0. **Download**: Baixa gabarito e cartões direto de uma pasta do Google Drive
1. **Processamento**: Extrai respostas usando visão computacional e clustering
2. **Cabeçalho**: Usa Google Gemini para extrair dados do aluno (nome, escola, turma, nascimento)
3. **Correção**: Compara respostas do aluno com o gabarito
4. **Resultados**: Envia automaticamente para Google Sheets com rate limiting

### 1. Clone o repositório

```bash
git clone https://github.com/JEAND1AS/cartao-resposta.git
cd cartao-resposta
```

### 2. Pré-requisitos para o passo de instalação
- **Python 3.8+**
- **Google Cloud APIs** (Sheets + Gemini)

## 🛠️ Instalação

###Criar e ativar ambiente virtual

#### Windows (PowerShell):

```bash

# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
.\.venv\Scripts\Activate.ps1

#Comando para instalar dependencias
pip install -r requirements.txt
```

#### Linux/macOS:
```bash
# Criar ambiente virtual
python3 -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate

#Comando para instalar dependencias
pip install -r requirements.txt
```


## ⚙️ Configuração

### 1. Configurar arquivo .env para guardar chaves secretas

- A biblioteca do .env será instalada automaticamente após executar o requirements.txt
- Dentro do .env defina os nomes das variáveis de ambiente ex: (GEMINI_API_KEY = sua_key_aqui, GOOGLE_SHEETS_ID = "sua_key_aqui", DRIVE_FOLDER_ID = "sua_key_aqui")


### 2. Google Sheets API

Siga as instruções em [`INSTRUCOES_GOOGLE_SHEETS.md`](INSTRUCOES_GOOGLE_SHEETS.md) para:
- Criar projeto no Google Cloud
- Ativar APIs necessárias  
- Gerar credenciais de service account
- Salvar como `credenciais_google.json`
- Jogar o `credenciais_google.json` dentro da pasta raiz

### 3. Google Gemini AI

Siga as instruções em [`GEMINI_SETUP.md`](GEMINI_SETUP.md) para:
- Obter API key do Gemini
- Configurar na variável de ambiente .env

### 4. Google Drive API

Para baixar os cartões direto do Google Drive:
- Ative também a **Google Drive API** no mesmo projeto
- Compartilhe a pasta (ou subpasta) do Drive com o e-mail da service account
- Copie o **ID da pasta** (ex.: `https://drive.google.com/drive/folders/ID_AQUI`)
- defina a variável de ambiente `DRIVE_FOLDER_ID` dentro do arquivo .env

### 5. Criar pastas e planilhas no Google Drive:

Crie com essa estrutura:

Google Drive
└── PastA "cartão-resposta" (Pasta raiz)
├── Pasta "5° ano" (44 questões) (Subpasta)
│ └── Planilha com informações dos alunos do 5° ano
├── Pasta "9° ano" (52 questões) (Subpasta)
│ └── Planilha com informações dos alunos do 9° ano
└── Arquivo "gabarito" (52 ou 44 questões) Voce precisa ter o gabarito nomeado como gabarito.jpg ou gabarito.png

OBS: Verifque o cabeçalho das planilhasr, está disponível dentro do README INSTRUCOES_GOOGLE_SHEETS

## 🎮 Como Usar

### Modo Local para ler os cartões disponível na pasta do drive

```bash
python script.py
```

### Modo monitor para ficar verificando a pasta do drive e ler automaticamente quando um novo cartão é adicionado

```bash
python script.py -monitor --intervalo 1
```



O sistema irá ler automaticamente a pasta `./gabaritos`,
processar todos os arquivos e enviar para o Google Sheets.

Fluxo completo:
1. Detectar automaticamente gabarito e alunos
2. Processar todos os cartões
3. Enviar resultados para Google Sheets
4. Mostrar relatório final



O script irá baixar todos os arquivos permitidos daquela pasta do Drive para
um diretório temporário, processar os cartões e remover os arquivos no final.



### Exemplo de Saída

```
📄 Enviando para planilha de 52 questões...
📊 Registro adicionado:
   🏫 Escola: E. M. João Francisco Braz
   👤 Aluno: Vitória Ferreira
   📅 Nascimento: 10/08/2010
   📚 Turma: 9° ano
   📊 Resultado: 16 acertos | 36 erros | 30.8%

📋 GABARITO DAS QUESTÕES:
==============================
1-D  2-C  3-A  4-D  5-C  6-A  7-A  8-C  9-A  10-D
11-A  12-A  13-B  14-D  15-B  16-D  17-A  18-A  19-A  20-D
21-A  22-C  23-D  24-A  25-A  26-D  27-B  28-C  29-D  30-A
31-B  32-B  33-B  34-C  35-C  36-B  37-A  38-D  39-C  40-D
41-D  42-A  43-B  44-C  45-C  46-B  47-A  48-B  49-C  50-D
51-C  52-D
==============================
```

### Customização no Código

```bash

## 📊 Formato do Google Sheets


| Data/Hora  | Escola | Aluno  | Nascimento | Turma | Acertos | Erros  | Questoes anuladas | Porcentagem |
|------------|--------|--------|------------|-------|---------|------- |-------------------|-------------|
| 25/09/2025 |   ABC  | João   | 15/03/2005 |  902  |    42   |    10  |        0          |   80.8%     |

## 🐛 Solução de Problemas

### Erro de OCR
```bash
# Verificar se Tesseract está instalado
tesseract --version

# No Windows, adicionar ao PATH:
# C:\Program Files\Tesseract-OCR
```

### Erro de Google Sheets
```bash
# Verificar se o arquivo de credenciais existe
ls credenciais_google.json

# Verificar se a planilha foi compartilhada com o service account
```

### Erro de Gemini
```bash
# Verificar se a API key está configurada
echo $GEMINI_API_KEY
```

### Baixa Precisão na Detecção
- Verificar qualidade das imagens (mínimo 300 DPI)
- Garantir boa iluminação e contraste
- Evitar sombras ou reflexos
- Usar modo debug para analisar detecções

