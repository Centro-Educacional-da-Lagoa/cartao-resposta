import axios from 'axios';

const DATA_API_URL = import.meta.env.VITE_DATA_API_URL || "http://localhost:3001";
const BOT_API_URL = import.meta.env.VITE_BOT_API_URL || "http://localhost:5000";

export const ANOS = ['4', '5', '8', '9'] as const;
export type Ano = typeof ANOS[number];
export type AnoFiltro = Ano | 'geral';

console.log('🔗 DATA_API_URL configurada:', DATA_API_URL + '/api/status');
console.log('🤖 BOT_API_URL configurada:', BOT_API_URL);

// 1. O Pulo do Gato: Configurar o Axios para enviar o header em TODAS as requisições automaticamente
axios.defaults.headers.common['ngrok-skip-browser-warning'] = 'true';
axios.defaults.withCredentials = true;

export interface Aluno {
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

export interface Estatisticas {
    status: string;
    ano: string;
    total_alunos: number;
    media_geral: number;
    nota_mais_alta: number;
    nota_mais_baixa: number;
    aprovados: number;
    reprovados: number;
}

export interface Status {
    status: 'idle' | 'running' | 'error';
    timestamp: string;
    bot_ativo: boolean;
    arquivo_atual: string | null;
    progresso: number;
    corrigidos_sessao: number;
    ultima_correcao_sessao: string | null;
    ultima_atualizacao: string | null;
    total_registros_4ano: number;
    total_registros_9ano: number;
    total_registros_5ano: number;
    total_registros_8ano: number;
    database: 'connected' | 'disconnected';
    vultr_s3?: {
        configured: boolean;
    };
}

export interface ArquivoStorage {
    id: string;
    name: string;
    mimeType: string;
    createdTime: string;
    size?: string;
}

export type BotStreamState = {
    status: 'idle' | 'running' | 'error';
    current_file: string | null;
    progress: number;
    total_corrected: number;
    last_correction: string | null;
    logs: { time: string; msg: string }[];
};

export interface PastaResponse {
    status: string;
    pasta: string;
    descricao: string;
    total_registros: number;
    arquivos: ArquivoStorage[];
}

export interface UploadResponse {
    status: string;
    message?: string;
    filename?: string;
    [key: string]: unknown;
}

export interface Usuario {
    id: string;
    nome: string;
    email: string;
    createdAt: string;
}

export interface AuthResponse {
    user: Usuario;
    accessToken: string;
    expiresIn: number;
}

export interface LoginPayload {
    email: string;
    password: string;
}

export interface RegisterPayload extends LoginPayload {
    nome: string;
}

export function conectarStreamBot(onData: (d: BotStreamState) => void, onError?: (e: Event) => void) {
    const es = new EventSource(BOT_API_URL + '/api/bot/stream');
    es.onmessage = (ev) => onData(JSON.parse(ev.data));
    if (onError) es.onerror = onError;
    return () => es.close();
}

export async function buscarStatus(): Promise<Status> {
    try {
        // 2. Colocamos o header no fetch também!
        const response = await fetch(`${DATA_API_URL}/api/status`, {
            credentials: 'include',
            headers: {
                'ngrok-skip-browser-warning': 'true'
            }
        });
        if (!response.ok) {
            throw new Error('Erro ao buscar status');
        }
        return await response.json();
    } catch (error) {
        console.error('Erro ao buscar status:', error);
        throw error;
    }
}

export const api = {
    async login(payload: LoginPayload): Promise<AuthResponse> {
        const response = await axios.post(`${DATA_API_URL}/api/auth/login`, payload);
        return response.data;
    },

    async registrar(payload: RegisterPayload): Promise<AuthResponse> {
        const response = await axios.post(`${DATA_API_URL}/api/auth/register`, payload);
        return response.data;
    },

    async loginGoogle(credential: string): Promise<AuthResponse> {
        const response = await axios.post(`${DATA_API_URL}/api/auth/google`, { credential });
        return response.data;
    },

    async getUsuarioAtual(): Promise<Usuario> {
        const response = await axios.get(`${DATA_API_URL}/api/auth/me`);
        return response.data;
    },

    async logout(): Promise<void> {
        await axios.post(`${DATA_API_URL}/api/auth/logout`);
    },

    async getStatus() {
        // Como configuramos o axios.defaults lá em cima, não precisa repetir o header aqui
        const response = await axios.get(`${DATA_API_URL}/api/status`);
        return response.data;
    },

    async getAlunos(ano: Ano) {
        const response = await axios.get(`${DATA_API_URL}/api/aluno/${ano}ano`);
        return response.data
    },

    async getEstatisticas(ano: Ano): Promise<Estatisticas> {
        const response = await axios.get(`${DATA_API_URL}/api/estatisticas/${ano}ano`);
        return response.data
    },

    async getEstatisticasGeral() {
        const response = await axios.get(`${DATA_API_URL}/api/estatisticas/geral`);
        return response.data
    },

    async getPasta(ano: Ano): Promise<PastaResponse> {
        const response = await axios.get(`${DATA_API_URL}/api/pasta/${ano}ano`);
        return response.data
    },

    async uploadArquivo(arquivo: File): Promise<UploadResponse> {
        const formData = new FormData();
        formData.append('arquivo', arquivo);

        const response = await axios.post(`${DATA_API_URL}/api/upload`, formData);
        return response.data;
    },
}
