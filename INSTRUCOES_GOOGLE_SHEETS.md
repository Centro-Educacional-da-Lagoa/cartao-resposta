# 🚀 INSTRUÇÕES PARA CONFIGURAR GOOGLE SHEETS

## PASSO 1: SUBSTITUIR CREDENCIAIS
1. Abra o arquivo baixado do Google Cloud Console
2. Copie todo o conteúdo JSON
3. Substitua o conteúdo do arquivo 'exemplo_credenciais.json'
4. Renomeie para 'credenciais_google.json'

## PASSO 2: CRIAR PLANILHA
1. Acesse: https://sheets.google.com
2. Crie nova planilha com nome: "Correção Cartão Resposta"
3. Compartilhe com o email da conta de serviço (encontrado no JSON)
4. Dê permissão de "Editor"

## PASSO 3: Configurar GOOGLE_SHEETS_ID
1. Baixe todas as depedências (bibliotecas) do python incluso nesse projeto 

NOTA: Se não fizer isso, o script criará automaticamente!

## PASSO 3: EXECUTAR O SCRIPT
1. Execute: python script.py
2. O script irá:
   ✅ Processar as imagens
   ✅ Mostrar resultados no terminal
   ✅ Conectar ao Google Sheets
   ✅ Enviar dados para planilha
   ✅ Perguntar se quer criar planilha detalhada

## ESTRUTURA DA PLANILHA:
- Data/Hora
- Escola, Aluno, Nascimento, Turma
- Total Questões, Acertos, Erros, Percentual
- Questões Acertadas, Questões Erradas

