# 🚀 INSTRUÇÕES PARA CONFIGURAR GOOGLE SHEETS E GOOGLE DRIVE

## PASSO 1: CRIAR E CONFIGURAR GOOGLE DRIVE
1. Criar uma pasta principal com o nome Cartao-resposta
2. Compartilhar a pasta com o email da conta de serviço e dar permissão de "Editor"
3. criar 2 pastas dentro da pasta principal (9° ano e 5° ano)
4. pegar o ID da pasta principal e das 2 subpastas
5. https://drive.google.com/drive/u/0/folders/xxsadsadwqdsa
                                              ↑ ESTE é o ID
5. configurar o ID dentro das aspas no .env de acordo com o nome.
    DRIVER_FOLDER_ID="Seu_id_aqui"
    DRIVER_FOLDER_9ANO="Seu_id_aqui"
    DRIVER_FOLDER_5ANO="Seu_id_aqui"

## PASSO 2: CRIAR PLANILHA DENTRO DAS SUBPASTAS 5° ANO E 9° ANO
1. Acesse as pastas
2. Crie as novas planilhas dentro dessas subpastas
5. Agora precisamos pegar o ID da planilha que será utilizada pelas turmas 5° ano e 9° ano para configurar dentro do .env
6. Acesse a planilha
7. https://docs.google.com/spreadsheets/d/1dsa12dsasa23/edit#gid=0
                                           ↑ ID da planilha


## ESTRUTURA ESPERADA DA PLANILHA:
```bash
## 📊 Formato do cabeçalho do Google Sheets

| Data/Hora  | Escola |  Aluno | Nascimento | Turma | Acertos Língua portuguesa | Acertos Matemática | Erros Lingua portuguesa | Erros Matemática | Questoes anuladas | Porcentagem |


```

