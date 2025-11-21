# 🚀 INSTRUÇÕES PARA CONFIGURAR GOOGLE SHEETS


## PASSO 1: CRIAR PLANILHA DENTRO DAS SUBPASTAS 5° ANO E 9° ANO
1. Acesse: https://sheets.google.com
2. Crie as novas planilhas dentro dessas subpastas - O nome é como você quiser
3. Compartilhe com o email da conta de serviço (encontrado no JSON)
4. Dê permissão de "Editor"
5. Agora precisamos pegar o ID da planilha que será utilizada pelas turmas 5° ano e 9° ano para configurar dentro do .env
6. Acesse a planilha e esse será a URL https://docs.google.com/spreadsheets/d/SEU_ID_AQUI/edit?gid=0#gid=0
7. Onde está escrito "seu_id_aqui" é uma sequencia de letras e números, esse será seu ID


## ESTRUTURA DA PLANILHA:
```bash
## 📊 Formato do Google Sheets

| Data/Hora  | Escola | Aluno  | Nascimento | Turma | Acertos | Erros  | Questoes anuladas | Porcentagem |
|------------|--------|--------|------------|-------|---------|------- |-------------------|-------------|
| 25/09/2025 |   ABC  | João   | 15/03/2005 |  902  |    42   |    10  |        0          |   80.8%     |

```

