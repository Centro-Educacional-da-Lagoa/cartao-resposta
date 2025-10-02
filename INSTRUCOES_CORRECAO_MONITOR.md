# 🤖 MODO MONITOR - SISTEMA DE PROCESSAMENTO INTELIGENTE

## 📋 VISÃO GERAL

O **Modo Monitor** é um sistema de processamento contínuo e automatizado de cartões-resposta que monitora uma pasta do Google Drive e processa automaticamente novos cartões assim que são adicionados.

---

## 🎯 COMO FUNCIONA

### **1. Sistema de Rastreamento por ID**

Cada arquivo no Google Drive possui um **ID único** (exemplo: `1MDeAOgDlbiScvf_6bmc7KkyvCsnlN44R`). O monitor usa esses IDs para rastrear quais arquivos já foram processados e criar um histórico_monitoramento.json com os ID

```
📁 Google Drive
├─ gabarito.png (ID: abc123...)     ← Nunca marcado como processado (NÃO REMOVA ESSE ARQUIVO)
├─ aluno_1.png (ID: def456...)      ← Processado? Sim → Ignora
├─ aluno_2.png (ID: ghi789...)      ← Processado? Não → Processa!
└─ aluno_3.png (ID: jkl012...)      ← Processado? Não → Processa!
```

### **2. Arquivo de Histórico**

O monitor mantém um arquivo `historico_monitoramento.json` que armazena:
- Lista de IDs já processados
- Data da última verificação
- Total de arquivos processados

```json
{
  "ultima_verificacao": "2025-10-02T10:30:00",
  "total_processados": 15,
  "arquivos_processados": [
    "1MDeAOgDlbiScvf_6bmc7KkyvCsnlN44R",
    "1XYZ789ABCdefGHI456jklMNO123pqr",
    ...
  ]
}
```

### **3. Ciclo de Verificação**

```
┌─────────────────────────────────────┐
│     🔍 Verificação #1 (10:00)       │
├─────────────────────────────────────┤
│  1. Listar arquivos no Drive        │
│  2. Comparar IDs com histórico      │
│  3. Identificar NOVOS cartões       │
│  4. Baixar APENAS os novos          │
│  5. Processar cada novo             │
│  6. Enviar para Google Sheets       │
│  7. Mover para pasta processados    │
│  8. Atualizar histórico             │
│  9. Aguardar intervalo              │
└─────────────────────────────────────┘
         ⏰ Aguarda X minutos
┌─────────────────────────────────────┐
│  🔍 Verificação #2 (10:05)          │
│  ... repete o processo ...          │
└─────────────────────────────────────┘
```

---

## 🚀 COMO USAR

### **Comando Básico**

```bash
python script.py --monitor --intervalo 5

```

- `--monitor`: Ativa o modo de monitoramento contínuo
- `--intervalo 5`: Verifica a cada 5 minutos (Você pode colocar qualquer número em minutos)
- `--Parar`: Caso queira parar, use o CTRL+C
---

## ⚙️ LÓGICA DE PROCESSAMENTO

### **Etapa 1: Verificação de Novos Arquivos**

```python
def verificar_novos_arquivos():
    """
    1. Lista TODOS os arquivos do Google Drive
    2. Carrega histórico de IDs processados
    3. Filtra apenas cartões NOVOS:
       - Não está no histórico (ID não processado)
       - Nome NÃO contém 'gabarito'
       - Tem extensão válida (.pdf, .png, .jpg, .jpeg)
    4. Retorna lista de novos + histórico atual
    """
```

**Resultado:**
```
🆕 Encontrados 2 NOVOS cartões!
```

### **Etapa 2: Processamento Seletivo**

```python
# Processa APENAS os novos
for cartao_novo in novos_cartoes:
    baixar(cartao_novo)
    processar(cartao_novo)
    comparar_com_gabarito(cartao_novo)
    enviar_para_sheets(cartao_novo)
    marcar_como_processado(cartao_novo['id'])
```

### **Etapa 3: Atualização de Histórico**

Após processar com sucesso:
```python
# Adicionar IDs dos novos processados ao histórico
arquivos_processados.update(['1ABC123...', '1XYZ789...'])

# Salvar histórico atualizado
salvar_historico(arquivos_processados)
```

---

## 📊 EXEMPLO PRÁTICO

### **Situação Inicial**

```
📁 Google Drive (Pasta Principal)
├─ gabarito.png
├─ aluno_1.png  ← Já processado
├─ aluno_2.png  ← Já processado
└─ aluno_3.png  ← Já processado

📝 historico_monitoramento.json
└─ 3 IDs registrados
```

### **Usuário adiciona 2 novos cartões**

```
📁 Google Drive (Pasta Principal)
├─ gabarito.png
├─ aluno_1.png  ← Já processado
├─ aluno_2.png  ← Já processado
├─ aluno_3.png  ← Já processado
├─ aluno_4.png  ← 🆕 NOVO!
└─ aluno_5.png  ← 🆕 NOVO!
```

### **Monitor detecta e processa**

```
🔍 Verificação #5 - 02/10/2025 10:25:00
🆕 Encontrados 2 NOVOS cartões!

🚀 Processando APENAS os novos cartões...


📋 Baixando gabarito e cartoes resposta em uma pasta provisória dentro do TEMP

📦 Movendo os cartões até a pasta cartoes-processados após processar todos...
📊 Novos processados: 2
📝 Total no histórico: 5
```

### **Resetar Histórico (Reprocessar Tudo)**

```bash
# Windows
del historico_monitoramento.json

# Linux/Mac
rm historico_monitoramento.json
```

Também podemos excluir manualmente o .json no diretório.

## � SOLUÇÃO DE PROBLEMAS

### **Problema: "Nenhum cartão novo encontrado" mas há arquivos**

**Causa:** Arquivos já estão no histórico

**Solução:**
```bash
# Ver histórico
python ver_ids_drive.py
```

Ou verificar o ID clicando em "copiar link" no arquivo

Será algo assim:

https://drive.google.com/file/d/1ekGgGDeaVMDv9RSBDjfN-qP3m7heRB_n/view?usp=drive_link

O ID é sempre depois do d/

**Por isso usamos IDs, não nomes!** 🎯
