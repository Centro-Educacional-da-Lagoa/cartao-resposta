# 🤖 CONFIGURAÇÃO DO GEMINI

## 📋 Pré-requisitos

### 1. Instalar Biblioteca do Gemini via ( pip install -r requirements.txt ) e caso não funcione, use o código abaixo.
```bash
pip install google-generativeai
```

### 2. Obter API Key do Gemini
1. Acesse: https://makersuite.google.com/app/apikey
2. Faça login com sua conta Google
3. Clique em "Create API Key"
4. Copie a chave gerada

### 3. Configurar API Key dentro do arquivo .env

```.env
GEMINI_API_KEY = "sua-chave-real-aqui"
```

## 🎯 Funcionalidades do Gemini principal do gemini
- Analisar o cabeçalho dos cartões-resposta e trazer as informações


## ⚠️ Considerações

### Erros de análise
- Cartões-resposta com cabeçalho em manuscrito podem conter erros
- Orientar cartões-resposta com o cabeçalho digitalizado com as informações dos alunos para maior confiabilidade
- OMR nunca é 100% preciso, por isso é recomendável as folhas estarem alinhadas, bem preenchidas com bola (não rabiscadas) e com uma boa iluminação (sem sombra sobre o papel)

### Custos
- Gemini API tem custo por uso
- Gratuito até certo limite mensal
- Veja preços em: https://ai.google.dev/pricing

### Internet
- Requer conexão ativa com internet
- Upload das imagens para análise

### Privacidade
- Imagens são enviadas para servidores Google
- Considere políticas de privacidade da instituição

### Erro: "API key inválida"
- Verifique se a API está correta no arquivo .env
- Verifique se a biblioteca do .env está instalada
- Certifique-se que não há espaços extras
- Gere nova chave se necessário

### Erro: "Quota exceeded"
- Limite gratuito atingido
- Configure pagamento ou aguarde reset mensal




