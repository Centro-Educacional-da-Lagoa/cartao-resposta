# 📝 Sistema Automatizado de Correção de Cartões Resposta

> **Sistema inteligente para correção automática de cartões resposta (gabaritos) usando OCR, visão computacional e IA**

## 🚀 Funcionalidades

- ✅ **Detecção automática** de gabaritos e folhas de resposta
- 🔍 **OCR avançado** com processamento de imagem otimizado  
- 🤖 **Extração de cabeçalho** com Google Gemini AI
- 📊 **Integração com Google Sheets** para armazenamento automático
- 🎯 **Alta precisão** na detecção de respostas marcadas
- 📁 **Processamento em lote** de múltiplos alunos
- 🔄 **Rate limiting** integrado para APIs
- 🐛 **Modo debug** com visualização detalhada
- 📱 **Suporte a PDF e imagens** (PNG, JPG, JPEG)

## 🎯 Como Funciona

1. **Detecção**: Identifica automaticamente o gabarito (arquivo com "gabarito" no nome)
2. **Processamento**: Extrai respostas usando visão computacional e clustering
3. **Cabeçalho**: Usa Google Gemini para extrair dados do aluno (nome, escola, turma, nascimento)
4. **Correção**: Compara respostas do aluno com o gabarito
5. **Resultados**: Envia automaticamente para Google Sheets com rate limiting

## 🛠️ Instalação

### Pré-requisitos

- **Python 3.8+**
- **Tesseract OCR**
- **Google Cloud APIs** (Sheets + Gemini)

### 1. Clone o repositório

```bash
git clone https://github.com/JEAND1AS/cartao-resposta.git
cd cartao-resposta
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Instale o Tesseract OCR

#### Windows
```bash
# Baixe de: https://github.com/UB-Mannheim/tesseract/wiki
# Ou use chocolatey:
choco install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-por
```

#### macOS
```bash
brew install tesseract
```

## ⚙️ Configuração

### 1. Google Sheets API

Siga as instruções em [`INSTRUCOES_GOOGLE_SHEETS.md`](INSTRUCOES_GOOGLE_SHEETS.md) para:
- Criar projeto no Google Cloud
- Ativar APIs necessárias  
- Gerar credenciais de service account
- Salvar como `credenciais_google.json`

### 2. Google Gemini AI

Siga as instruções em [`GEMINI_SETUP.md`](GEMINI_SETUP.md) para:
- Obter API key do Gemini
- Configurar variáveis de ambiente

### 3. Estrutura de pastas

```
cartao-resposta/
├── gabaritos/              # Pasta com gabarito + cartões dos alunos
│   ├── gabarito.png        # Gabarito (nome deve começar com "gabarito")
│   ├── aluno1.png          # Cartões dos alunos
│   ├── aluno2.jpg          # Aceita PNG, JPG, JPEG, PDF
│   └── ...
├── script.py               # Script principal
├── credenciais_google.json # Suas credenciais (não incluído no git)
└── ...
```

## 🎮 Como Usar

### Modo Automático (Recomendado)

```bash
python script.py
```

O sistema irá:
1. Detectar automaticamente gabarito e alunos
2. Processar todos os cartões
3. Enviar resultados para Google Sheets
4. Mostrar relatório final

### Exemplo de Saída

```
✅ Gabarito identificado: gabarito.png
👥 Encontrados 5 alunos para processar

🔄 [01/5] Processando: aluno1.png
✅ Dados extraídos: JOÃO DA SILVA (Escola ABC)
✅ Respostas processadas: 52/52 questões detectadas

🔍 DEBUG - RESPOSTAS DETECTADAS PARA JOÃO DA SILVA:
============================================================
   Q01:A   Q02:C   Q03:A   Q04:C
   Q05:C   Q06:B   Q07:D   Q08:A
   [...]

📊 ESTATÍSTICAS DE DETECÇÃO:
   🅰️ Alternativa A: 14 questões
   🅱️ Alternativa B: 11 questões  
   🅲 Alternativa C: 20 questões
   🅳 Alternativa D: 7 questões
   ✅ Total detectado: 52/52 questões (100.0%)

📊 Resultado: 42/52 acertos (80.8%)
✅ Enviado para Google Sheets (1/5)

🎉 PROCESSAMENTO CONCLUÍDO!
📊 5 alunos processados com sucesso
```

## 🔧 Configurações Avançadas

### Variáveis de Ambiente

```bash
# Para Gemini AI
export GEMINI_API_KEY="sua_api_key_aqui"

# Para Google Drive (futuro)
export DRIVE_FOLDER_ID="id_da_pasta_drive"
```

### Customização no Código

```python
# Configurações principais no script.py
debug_mode = True          # Ativar debug detalhado
usar_gemini = True         # Extrair cabeçalho com IA
usar_google_sheets = True  # Enviar para planilha
```

## 📊 Formato do Google Sheets

O sistema cria/atualiza uma planilha com as colunas:

| Data/Hora | Escola | Aluno | Nascimento | Turma | Acertos | Erros | Percentual | Questões Detectadas |
|-----------|--------|--------|------------|-------|---------|-------|------------|-------------------|
| 25/09/2025 10:30 | Escola ABC | João Silva | 15/03/2005 | 3º A | 42 | 10 | 80.8% | 52 |

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

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📋 Roadmap

- [ ] Interface web com Flask/Django
- [ ] Monitoramento automático de pasta no Google Drive
- [ ] Suporte a diferentes layouts de cartão resposta
- [ ] API REST para integração
- [ ] Dashboard com estatísticas avançadas
- [ ] Processamento em paralelo
- [ ] Notificações por email/WhatsApp

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

**Jean Dias**
- GitHub: [@JEAND1AS](https://github.com/JEAND1AS)
- Email: [seu.email@exemplo.com]

## 🙏 Agradecimentos

- Google Cloud Platform (Sheets API + Gemini AI)
- OpenCV community
- Tesseract OCR developers
- Scikit-learn team

---

**⚡ Desenvolvido para tornar a correção de provas mais rápida, precisa e automatizada!**