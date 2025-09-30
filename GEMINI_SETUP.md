# 🤖 CONFIGURAÇÃO DO GEMINI

## 📋 Pré-requisitos

### 1. Instalar Biblioteca do Gemini
```bash
pip install google-generativeai
```

### 2. Obter API Key do Gemini
1. Acesse: https://makersuite.google.com/app/apikey
2. Faça login com sua conta Google
3. Clique em "Create API Key"
4. Copie a chave gerada

### 3. Configurar API Key
No arquivo `script.py`, encontre a linha:
```python
GEMINI_API_KEY = "SUA_API_KEY_AQUI"
```

Substitua por sua chave real:
```python
GEMINI_API_KEY = "sua-chave-real-aqui"
```

## 🚀 Como Usar

### Executar com Gemini
```bash
python script.py
```

O sistema perguntará:
- `🔧 MODO DEBUG - Ativar detecção avançada? (s/n):`
- `🤖 GEMINI - Ativar análise inteligente com Gemini? (s/n):`

Digite `s` para ativar o Gemini.

## 🎯 Funcionalidades do Gemini

### 1. Análise Inteligente de Imagens
- Usa Gemini Vision para analisar cartões resposta
- Identifica alternativas marcadas com precisão
- Ignora marcações de correção (círculos verdes)

### 2. Validação Cruzada
- Compara resultados OMR vs Gemini
- Escolhe automaticamente o melhor resultado
- Gera relatório de concordância

### 3. Correção Automática
- **Alta concordância (≥80%)**: Usa OMR
- **Média concordância (50-79%)**: Usa híbrido OMR/Gemini
- **Baixa concordância (<50%)**: Usa Gemini

## 📊 Benefícios

### Precisão Melhorada
- Reduz falsos positivos do OMR
- Detecta melhor alternativas pintadas vs não pintadas
- Ignora marcações de professores

### Confiabilidade
- Dupla validação (OMR + IA)
- Relatório de concordância entre métodos
- Fallback automático se um método falhar

### Flexibilidade
- Pode ser ativado/desativado facilmente
- Funciona mesmo se Gemini não estiver disponível
- Integração transparente com sistema existente

## ⚠️ Considerações

### Custos
- Gemini API tem custo por uso
- Gratuito até certo limite mensal
- Veja preços em: https://ai.google.dev/pricing

### Internet
- Requer conexão ativa com internet
- Upload das imagens para análise
- Processo mais lento que OMR local

### Privacidade
- Imagens são enviadas para servidores Google
- Considere políticas de privacidade da instituição
- Para dados sensíveis, use apenas modo OMR local

## 🔧 Solução de Problemas

### Erro: "google.generativeai not found"
```bash
pip install google-generativeai
```

### Erro: "API key inválida"
- Verifique se copiou a chave corretamente
- Certifique-se que não há espaços extras
- Gere nova chave se necessário

### Erro: "Quota exceeded"
- Limite gratuito atingido
- Configure pagamento ou aguarde reset mensal
- Use apenas modo OMR temporariamente

### Baixa concordância OMR vs Gemini
- Normal em imagens com qualidade ruim
- Verifique se imagens estão nítidas
- Considere melhorar iluminação/scan


