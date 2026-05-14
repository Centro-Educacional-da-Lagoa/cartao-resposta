# 📝 Sistema Automatizado de Correção de Cartões Resposta

> **Sistema inteligente para correção automática de cartões resposta (gabaritos) usando OCR, visão computacional e IA**

## 🚀 Funcionalidades

- ✅ **Detecção automática** de gabaritos e folhas de resposta
- 🤖 **Extração de cabeçalho** com Google Gemini AI
- 📊 **Integração com Google Sheets** para armazenamento automático
- 🗄️ **Integração com backend NestJS + PostgreSQL** para persistência oficial
- ☁️ **Sincronização com Google Drive** (download automático da pasta configurada)
- 🎯 **Alta precisão** na detecção de respostas marcadas
- 📁 **Processamento em lote** de múltiplos alunos
- 🔄 **Rate limiting** integrado para APIs
- 📐 **Correção de perspectiva automática condicional** (estilo scanner)
- 📍 **Recorte da área de respostas por quadrados de alinhamento**
- 🧹 **Filtros extras contra ruídos em cartões de 44 questões**
- 🐛 **Modo debug** com visualização detalhada
- 📱 **Suporte a PDF e imagens** (PNG, JPG, JPEG)

## 🎯 Como Funciona

0. **Download**: Baixa gabarito e cartões direto de uma pasta do Google Drive
1. **Processamento**: Extrai respostas usando visão computacional e clustering
2. **Cabeçalho**: Usa Google Gemini para extrair dados do aluno (nome, escola, turma, nascimento)
3. **Correção**: Compara respostas do aluno com o gabarito
4. **Resultados**: Envia automaticamente para Google Sheets e, quando configurado, para o backend

### 1. Clone o repositório

```bash
git clone https://github.com/JEAND1AS/cartao-resposta.git
cd cartao-resposta
```

### 2. Pré-requisitos para o passo de instalação
- **Python 3.8+**
- **Google Cloud APIs** (Drive, Sheets + Gemini)

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

### 1: CONFIGURAR GOOGLE CLOUD, API's, CONTA DE SERVIÇO E CREDENCIAIS_GOOGLE.JSON
- Acesse https://console.cloud.google.com/
- Criar um novo projeto
- Ativar API's do Google Sheets, Google Drive e Gemini for Google Cloud API
- Criar uma credencial de conta de serviços
- Marca a caixa do email criado e clicar em "Contas de Serviço"
- Criar uma nova chave JSON
- Irá baixar o arquivo.json, renoemar para ccredenciais_google.json e colocar dentro da pasta raiz

### 2. Criar o arquivo .env e configurar com os seguintes nomes:
   GEMINI_API_KEY="Sua_key_aqui"
   GOOGLE_SHEETS_9ANO="Sua_key_aqui"
   GOOGLE_SHEETS_5ANO="Sua_key_aqui"
   DRIVER_FOLDER_ID="Sua_key_aqui"
   DRIVER_FOLDER_9ANO="Sua_key_aqui"
   DRIVER_FOLDER_5ANO="Sua_key_aqui"
   GOOGLE_CREDENTIALS_JSON='{"type":"service_account","project_id":"...","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"...","client_id":"...","auth_uri":"...","token_uri":"...","auth_provider_x509_cert_url":"...","client_x509_cert_url":"...", "universe_domain":"googleapis.com"}'

### 3. (Opcional) Sincronizar com o banco via backend NestJS

Adicione no `.env`:

```env
BACKEND_SYNC_ENABLED=true
BACKEND_BASE_URL=http://localhost:3001
BACKEND_USER_EMAIL=admin@example.com
BACKEND_USER_PASSWORD=Admin@1234
BACKEND_SYNC_TIMEOUT_SECONDS=20
```

Com isso, cada cartão processado também é enviado para:

- `POST /api/v1/cartao-resposta/leituras`

Observações:

- O usuário do backend precisa da permissão `CartoesResposta.create`.
- O envio pode usar `turmaId/alunoId` (se disponíveis) ou `turmaNome/alunoNome` para resolução por matrícula.

OBS: A biblioteca do .env será instalada automaticamente após executar o requirements.txt



### 2. Google Drive e Google Sheets API

Siga as instruções em [`INSTRUCOES_GOOGLE_SHEETS.md`](INSTRUCOES_GOOGLE_SHEETS.md) para:
- Configurar o Google Drive
- Configurar as Planilhas

OBS: Verifque o cabeçalho das planilhas, está disponível dentro do README INSTRUCOES_GOOGLE_SHEETS

### 3. Google Gemini AI

Siga as instruções em [`GEMINI_SETUP.md`](GEMINI_SETUP.md) para:
- Obter API key do Gemini
- Configurar na variável de ambiente .env


## 📦 Dependências do Sistema

Além das bibliotecas Python (instaladas via `pip install -r requirements.txt`), você precisa instalar:

### 1. Tesseract OCR (Fallback caso Gemini falhe)
**Nota:** O sistema usa Gemini AI como método principal. O Tesseract OCR é apenas um fallback automático.

#### Windows:
```bash
# Via Chocolatey (recomendado)
choco install tesseract

# OU baixar manualmente:
# https://github.com/UB-Mannheim/tesseract/wiki
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-por
```

#### macOS:
```bash
brew install tesseract tesseract-lang
```

### 2. Poppler (Necessário para processar PDFs)

#### Windows:
```bash
# Via Chocolatey (recomendado)
choco install poppler

# Via Scoop (alternativa)
scoop install poppler

# OU manualmente:
# 1. Baixe: https://github.com/oschwartz10612/poppler-windows/releases/
# 2. Extraia para C:\poppler
# 3. Adicione C:\poppler\Library\bin ao PATH do sistema
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

#### macOS:
```bash
brew install poppler
```

### 3. Verificar Instalações

```bash
# Verificar Poppler
python -c "from pdf2image import convert_from_path; print('✅ Poppler OK!')"

# Verificar Tesseract (se instalado)
tesseract --version
```

### 📝 Notas Importantes

- **Poppler é obrigatório** para processar arquivos PDF
- **Tesseract é opcional** - usado apenas como fallback se Gemini falhar
- No Windows, se não tiver Chocolatey ou Scoop, use instalação manual e configure o PATH



## 🎮 Como Usar

### Modo monitor para ler de forma contínua e automática os gabaritos e cartões-resposta dentro da pasta

OBS: No modo Monitor, o sistema cria automaticamente o arquivo historico_monitoramento.json. Nesse arquivo são salvos os IDs de todos os cartões que já foram lidos, garantindo que o bot não leia o mesmo cartão mais de uma vez.

ATENÇÃO: Se você apagar esse arquivo ou o ID, o bot vai considerar que nenhum cartão foi lido ainda, e poderá ler todos novamente.

```bash
python script.py --monitor --intervalo 1
```

### 🧭 Correção automática de perspectiva (v1)

O pré-processamento agora aplica, por padrão:

1. Retificação de perspectiva condicional (4 pontos, estilo scanner)
2. Correção de rotação (deskew)

Se a confiança geométrica da perspectiva não for suficiente, o sistema **não força warp** e segue com deskew.

Para desativar a perspectiva (rollback operacional), use:

```bash
python script.py --monitor --intervalo 1 --no-perspectiva
```

### 🧪 Teste visual sob demanda (folhas tortas vs ajustadas)

Use o script abaixo para gerar inspeção visual em lote:

```bash
python teste_visual_perspectiva.py --input <pasta_imagens> --output <pasta_debug>
```

Saída gerada por imagem:

- `01_original`: imagem original
- `02_cantos_detectados`: contorno/cantos detectados
- `03_retificada`: imagem após warp de perspectiva
- `04_comparativo`: lado a lado (original vs retificada)
- `05_normalizada_final`: resultado final após cadeia completa (perspectiva + deskew)

Também é gerado `resumo_processamento.txt` na pasta de saída com status por arquivo:
`applied`, `ignored` ou `fallback`.

### 🆕 Atualizações recentes no processamento

O processamento de OMR recebeu ajustes para deixar a leitura dos gabaritos mais estável:

- **Quadrados de alinhamento**: o sistema tenta localizar os 4 quadrados pretos ao redor da área de respostas. Quando encontra, exibe `ROI por marcadores: warp=... | crop=...` e usa esses pontos para corrigir a perspectiva e recortar somente as alternativas.
- **Fallback seguro**: se os quadrados não forem encontrados, o bot não força a perspectiva. Ele mantém o fluxo legado com crop proporcional e registra no debug que os marcadores não foram localizados.
- **Marcadores ignorados como bolha**: os quadrados de alinhamento detectados nas bordas são descartados durante a leitura para não entrarem como respostas marcadas.
- **Cartões de 44 questões**: foi adicionado um filtro específico para reduzir ruídos de números, letras e linhas da tabela. Esse filtro ignora a faixa dos números das questões dentro de cada coluna e exige tamanho, preenchimento e circularidade compatíveis com uma bolha preenchida.
- **Cartões de 52 questões**: o crop proporcional de fallback foi ajustado para preservar melhor a borda direita quando o cartão não usa marcadores confiáveis.
- **Debug no monitor**: quando o modo debug está ativo, o processamento de gabaritos, PDFs e imagens individuais propaga esse modo para facilitar a inspeção de perspectiva, OCR e OMR.
- **Sincronização com backend**: além do Google Sheets, o resultado pode ser enviado ao backend configurado no `.env`, incluindo respostas normalizadas, gabarito, dados do aluno e resumo de acertos/erros.



O sistema irá ler automaticamente a pasta `Cartao-resposta`,
processar todos os arquivos e enviar para o Google Sheets.

Fluxo completo:
1. Jogar todos os arquivos dentro da pasta principal do bot
2. Detectar automaticamente gabaritos e alunos, podendo jogar 2 gabaritos de uma vez e todos os cartões-respostas
2. Processa todos os cartões do 5° e 9° ano via IA e OCR
3. Enviar resultados para Google Sheets e pastas de acordo com o ano detectado

OBS: O gabarito sempre tem que ter os seguintes critérios
gabarito44.jpg | gabarito_44.jpg | gabarito44.png | gabarito_44.png
gabarito52.jpg | gabarito_52.jpg | gabarito52.png | gabarito_52.png



O script irá baixar todos os arquivos permitidos daquela pasta do Drive para um diretório temporário, processar os cartões e remover os arquivos no final.



### Exemplo de Saída

```
PREPROCESSANDO ARQUIVO ALUNO_2: alanteste44_pb.jpeg
   ✅ Gemini detectou: 5° ano (44 questões)
   ✅ Gemini detectou com sucesso: 44 questões
   🔍 DEBUG - Dados extraídos: Escola=Col Intercultural School, Aluno=ALAN Oliveira Santos, Turma=501, Nasc=31/01/2019
   📁 Destino: Pasta 5° ano (44 questões)

────────────────────────────────────────────────────────────
👤 ALAN Oliveira Santos
📚 Turma: 501 | Escola: Col Intercultural School
✅ Acertos: 16
❌ Erros: 27
📊 Percentual: 37.2%

📝 Respostas:

📋 GABARITO DAS QUESTÕES:
==============================
1-a  2-b  3-a  4-c  5-c  6-d  7-b  8-d  9-c  10-a
11-d  12-a  13-b  14-c  15-d  16-c  17-b  18-a  19-b  20-b
21-c  22-b  23-a  24-a  25-b  26-a  27-b  28-c  29-b  30-d
31-c  32-d  33-b  34-b  35-a  36-b  37-c  38-c  39-c  40-b
41-a  42-b  43-d  44-c
==============================
────────────────────────────────────────────────────────────
📄 Enviando para planilha de 44 questões...
📊 Registro adicionado:
   🏫 Escola: col intercultural school
   👤 Aluno: alan oliveira santos
   📅 Nascimento: 31/01/2019
   📚 Turma: 501
   📊 Resultado: ✓ 11PT/5MT | ✗ 11PT/16MT | 1 anuladas | 37.2%
```

### Customização no Código

```bash

## 📊 Formato do cabeçalho do Google Sheets


| Data/Hora  | Escola |  Aluno | Nascimento | Turma | Acertos Língua portuguesa | Acertos Matemática | Erros Lingua portuguesa | Erros Matemática  | Questoes anuladas | Porcentagem |


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

