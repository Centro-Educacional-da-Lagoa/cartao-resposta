export type StatusResponse = {
  status: 'idle' | 'running' | 'error';
  timestamp: string;
  bot_ativo: boolean;
  arquivo_atual: string | null;
  progresso: number;
  corrigidos_sessao: number;
  ultima_correcao_sessao: string | null;
  ultima_atualizacao: string | null;
  total_registros_9ano: number;
  total_registros_5ano: number;
  database: 'connected' | 'disconnected';
  google_drive: {
    configured: boolean;
    folder_9ano: boolean;
    folder_5ano: boolean;
  };
};
