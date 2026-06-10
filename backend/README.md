# Backend NestJS - Resultados Escolares

API em NestJS com Prisma para armazenar e consultar resultados de alunos do 5o e 9o ano.

## Subir localmente

```bash
cd backend
cp .env.example .env
npm install
npm run prisma:generate
npm run prisma:db-push
npm run start:dev
```

Por padrao a API sobe em `http://localhost:3001`.

## Endpoints principais

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/google`
- `GET /api/auth/me`
- `POST /api/auth/logout`
- `GET /api/aluno/4ano`
- `GET /api/aluno/5ano`
- `GET /api/aluno/8ano`
- `GET /api/aluno/9ano`
- `GET /api/estatisticas/4ano`
- `GET /api/estatisticas/5ano`
- `GET /api/estatisticas/8ano`
- `GET /api/estatisticas/9ano`
- `GET /api/estatisticas/geral`
- `GET /api/pasta/4ano`
- `GET /api/pasta/5ano`
- `GET /api/pasta/8ano`
- `GET /api/pasta/9ano`
- `GET /api/status`

Tambem existem endpoints CRUD para gerenciamento dos resultados:

- `POST /api/aluno`
- `GET /api/aluno`
- `GET /api/aluno/:id`
- `PATCH /api/aluno/:id`
- `DELETE /api/aluno/:id`

As consultas do painel exigem sessao autenticada. O endpoint `POST /api/aluno` continua publico para manter a ingestao do worker Python, mas tambem aceita `Authorization: Bearer <token>` quando `BACKEND_AUTH_ENABLED=true` estiver configurado no worker.

## Autenticacao

Configure as variaveis abaixo para login local, JWT e Google Sign-In:

```env
JWT_SECRET=gere-um-segredo-com-pelo-menos-32-caracteres
JWT_EXPIRES_IN_SECONDS=604800
AUTH_COOKIE_SECURE=false
GOOGLE_OAUTH_CLIENT_ID=seu-client-id.apps.googleusercontent.com
```

Use o mesmo client ID no frontend como `VITE_GOOGLE_CLIENT_ID`. Senhas sao gravadas na coluna `senha` como hash `scrypt`; contas criadas pelo Google ficam com `senha` nula ate o usuario definir uma senha local.

## Banco de dados

Configure `DATABASE_URL` para SQL Server. Exemplo com SQL Server Management Studio e o Compose raiz:

```env
DATABASE_URL="sqlserver://localhost:1433;database=cartao_resposta;user=sa;password=CartaoResposta%402026;encrypt=false;trustServerCertificate=true"
```

Se precisar manter compatibilidade com as variaveis antigas, o backend ainda aceita `DB_HOST`, `DB_PORT`, `DB_USERNAME`, `DB_PASSWORD`, `DB_DATABASE`, `DB_ENCRYPT` e `DB_TRUST_SERVER_CERTIFICATE` quando `DATABASE_URL` nao estiver definido.

Para manter o schema sincronizado localmente:

```bash
npm run prisma:db-push
```
