# 📊 Sistema de Correção de Cartões - Frontend

Dashboard web interativo para visualização e análise de dados de desempenho de alunos do Centro Educacional da Lagoa. Este sistema fornece estatísticas detalhadas, gráficos interativos e ferramentas de exportação para dados dos alunos de 5º e 9º ano.

## 🎯 Sobre o Projeto

O Sistema de Correção de Cartões é uma aplicação frontend desenvolvida em React + TypeScript que consome dados de uma API backend, exibindo informações acadêmicas de forma visual e intuitiva. O sistema permite análise de desempenho em Língua Portuguesa e Matemática, com recursos avançados de filtragem e exportação.

### Funcionalidades Principais

- 📈 Dashboard com estatísticas em tempo real
- 📊 Gráficos interativos (Pizza, Barras e Linha)
- 🔍 Sistema de filtros avançados (nome, escola, turma, período)
- 📄 Exportação de relatórios em PDF e Excel
- 🎨 Interface moderna e responsiva
- 📱 Design adaptativo para dispositivos móveis
- 🔄 Navegação entre 5º ano, 9º ano e visão geral

## 🛠️ Tecnologias Utilizadas

- **React 19.2.0** - Biblioteca principal
- **TypeScript** - Tipagem estática
- **Vite** - Build tool e dev server
- **Tailwind CSS** - Estilização
- **Axios** - Requisições HTTP
- **Recharts** - Gráficos interativos
- **jsPDF** - Geração de PDF
- **XLSX** - Exportação para Excel
- **Lucide React** - Ícones modernos

## 📁 Estrutura do Projeto

```
src/
├── components/
│   ├── Dashboard.tsx           # Componente principal do dashboard
│   ├── CardsEstatisticas.tsx   # Cards com métricas (total, média, aprovação)
│   ├── Graficos.tsx            # Gráficos de pizza, barras e linha
│   ├── Filtros.tsx             # Sistema de filtros avançados
│   ├── TabelaAlunos.tsx        # Tabela com listagem de alunos
│   └── BotoesExportacao.tsx    # Botões para exportar PDF/Excel
├── service/
│   └── api.ts                  # Configuração e chamadas da API
├── App.tsx                     # Componente raiz da aplicação
├── main.tsx                    # Ponto de entrada
└── index.css                   # Estilos globais
```

## 🧩 Componentes

### 📊 Dashboard
Componente central que gerencia o estado da aplicação e coordena todos os subcomponentes.

**Responsabilidades:**
- Carrega dados dos alunos via API
- Gerencia estado de carregamento
- Controla filtros aplicados
- Distribui dados para componentes filhos

**Props:**
- `ano`: '4' | '5' | '8' | '9' | 'geral' - Define qual conjunto de dados exibir

### 📈 CardsEstatisticas
Exibe cards com estatísticas principais em formato visual atraente.

**Métricas exibidas:**
- Total de alunos
- Média geral de notas
- Taxa de aprovação/reprovação
- Nota mais alta e mais baixa

**Features:**
- Animações hover
- Ícones ilustrativos (Lucide React)
- Cores dinâmicas por categoria
- Cálculos em tempo real com `useMemo`

### 📊 Graficos
Renderiza visualizações gráficas dos dados usando Recharts.

**Tipos de gráficos:**
1. **Pizza** - Proporção aprovados vs reprovados
2. **Barras** - Alunos e média por turma
3. **Linha** - Evolução temporal das médias
4. **Barras Múltiplas** - Comparação Português vs Matemática

**Features:**
- Responsivos e interativos
- Tooltips informativos
- Legendas personalizadas
- Cores temáticas

### 🔍 Filtros
Sistema de filtragem avançado com interface expansível.

**Campos de filtro:**
- Busca por nome
- Escola
- Turma
- Data início
- Data fim

**Features:**
- Contador de filtros ativos
- Botão de limpar filtros
- Filtros em tempo real
- Interface colapsável

### 📝 TabelaAlunos
Tabela completa com dados detalhados dos alunos.

**Colunas:**
- Data da avaliação
- Nome completo
- Escola
- Turma
- Porcentagem (nota)
- Acertos em Português
- Acertos em Matemática
- Erros em Português
- Erros em Matemática
- Questões anuladas

**Features:**
- Ordenação por data (mais recente primeiro)
- Filtros integrados
- Indicador visual de aprovação/reprovação
- Responsiva com scroll horizontal
- Integração com exportação

### 📄 BotoesExportacao
Componente para exportar dados em diferentes formatos.

**Formatos suportados:**
- **PDF** - Relatório formatado com jsPDF
- **Excel** - Planilha compatível com XLSX

**Características do PDF:**
- Cabeçalho institucional
- Data de geração
- Tabela formatada com autoTable
- Total de alunos
- Layout profissional

**Características do Excel:**
- Cabeçalhos formatados
- Dados organizados em colunas
- Nome do arquivo com ano e data

## 🔌 API Service

### Configuração
```typescript
const DATA_API_URL = import.meta.env.VITE_DATA_API_URL;
const BOT_API_URL = import.meta.env.VITE_BOT_API_URL;
```

### Interfaces TypeScript

#### Aluno
```typescript
interface Aluno {
    "Nome completo": string;
    Escola: string;
    Turma: string;
    Porcentagem: string;
    DATA: string;
    "Acertos Língua Portuguesa": string;
    "Acertos Matemática": string;
    "Erros Língua Portuguesa": string;
    "Erros Matemática": string;
    "Questões anuladas": string;
}
```

#### Estatisticas
```typescript
interface Estatisticas {
    status: string;
    ano: string;
    total_alunos: number;
    media_geral: number;
    nota_mais_alta: number;
    nota_mais_baixa: number;
    aprovados: number;
    reprovados: number;
}
```

#### Status
```typescript
interface Status {
    status: string;
    timestamp: Date;
    "bot_ativo": boolean;
    "ultima_atualizacao": string;
    "total_registros_9ano": number;
    "total_registros_5ano": number;
}
```

### Endpoints da API

```typescript
// Status do sistema
api.getStatus()
// GET /api/status

// Alunos do 9º ano
api.getAlunos('9')
// GET /api/aluno/9ano
// Retorna: { alunos: Aluno[] }

// Alunos do 5º ano
api.getAlunos('5')
// GET /api/aluno/5ano
// Retorna: { alunos: Aluno[] }

// Estatísticas do 9º ano
api.getEstatisticas('9')
// GET /api/estatisticas/9ano
// Retorna: Estatisticas

// Estatísticas do 5º ano
api.getEstatisticas('5')
// GET /api/estatisticas/5ano
// Retorna: Estatisticas

// Estatísticas gerais
api.getEstatisticasGeral()
// GET /api/estatisticas/geral
```

## 🚀 Instalação e Execução

### Pré-requisitos
- Node.js (versão 18 ou superior)
- npm ou yarn
- Backend NestJS/SQL Server rodando em `http://localhost:3001`
- API Python do bot rodando em `http://localhost:5000`, se as ações do bot forem usadas

### Instalação

```bash
# Clone o repositório
git clone https://github.com/JEAND1AS/frontend-bot.git

# Entre no diretório
cd frontend-bot

# Instale as dependências
npm install
```

### Executar em Desenvolvimento

```bash
npm run dev
```

O aplicativo estará disponível em `http://localhost:5173`

### Build para Produção

```bash
npm run build
```

### Preview da Build

```bash
npm run preview
```

### Lint

```bash
npm run lint
```

## 🎨 Estilização

O projeto utiliza **Tailwind CSS** para estilização, proporcionando:

- Design system consistente
- Classes utilitárias
- Responsividade mobile-first
- Gradientes e animações
- Temas customizáveis

### Configuração Tailwind
- `tailwind.config.js` - Configuração principal
- `postcss.config.js` - Processamento CSS
- `index.css` - Diretivas Tailwind

## 📊 Fluxo de Dados

```
App.tsx
  ↓ (ano selecionado)
Dashboard.tsx
  ↓ (carrega dados da API)
api.ts → Backend
  ↓ (retorna dados)
Dashboard.tsx
  ↓ (distribui dados)
├─→ CardsEstatisticas.tsx (métricas)
├─→ Graficos.tsx (visualizações)
├─→ Filtros.tsx (controles)
└─→ TabelaAlunos.tsx (listagem)
      └─→ BotoesExportacao.tsx (relatórios)
```

## 🔐 Segurança e Boas Práticas

- TypeScript para type safety
- Validação de dados da API
- Tratamento de erros com try/catch
- Uso de useMemo para otimização
- Componentização modular
- Código limpo e documentado

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto é privado e pertence ao Centro Educacional da Lagoa.

## 👥 Autor

**JEAND1AS**
- GitHub: [@JEAND1AS](https://github.com/JEAND1AS)

## 📧 Suporte

Para questões e suporte, entre em contato com a equipe de desenvolvimento.

---

© 2026 Centro Educacional da Lagoa - Todos os direitos reservados
