export type StatusResponse = {
  status: 'idle' | 'running' | 'error';
  timestamp: string;
  bot_ativo: boolean;
  arquivo_atual: string | null;
  progresso: number;
  corrigidos_sessao: number;
  ultima_correcao_sessao: string | null;
  ultima_atualizacao: string | null;
  total_registros_4ano: number;
  total_registros_5ano: number;
  total_registros_8ano: number;
  total_registros_9ano: number;
  database: 'connected' | 'disconnected';
  vultr_s3: {
    configured: boolean;
  };
};
