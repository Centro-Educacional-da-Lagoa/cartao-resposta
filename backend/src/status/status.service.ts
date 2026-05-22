import { Injectable, Logger } from '@nestjs/common';

import { AnoEscolar } from '../aluno/ano-escolar.enum';
import { AlunoService } from '../aluno/aluno.service';
import { GoogleDriveService } from '../google-drive/google-drive.service';
import { StatusResponse } from './status.types';

@Injectable()
export class StatusService {
  private readonly logger = new Logger(StatusService.name);

  constructor(
    private readonly alunoService: AlunoService,
    private readonly googleDriveService: GoogleDriveService,
  ) {}

  async getStatus(): Promise<StatusResponse> {
    const googleDriveStatus = {
      configured: this.googleDriveService.isConfigured(),
      folder_9ano: this.googleDriveService.isConfigured(AnoEscolar.NONO_ANO),
      folder_5ano: this.googleDriveService.isConfigured(AnoEscolar.QUINTO_ANO),
    };

    try {
      const [totalNonoAno, totalQuintoAno, latest] = await Promise.all([
        this.alunoService.countByAno(AnoEscolar.NONO_ANO),
        this.alunoService.countByAno(AnoEscolar.QUINTO_ANO),
        this.alunoService.findLatest(),
      ]);

      return {
        status: 'idle',
        timestamp: new Date().toISOString(),
        bot_ativo: false,
        arquivo_atual: null,
        progresso: 0,
        corrigidos_sessao: 0,
        ultima_correcao_sessao: null,
        ultima_atualizacao: latest ? this.alunoService.formatDatePtBr(latest.data) : null,
        total_registros_9ano: totalNonoAno,
        total_registros_5ano: totalQuintoAno,
        database: 'connected',
        google_drive: googleDriveStatus,
      };
    } catch (error) {
      this.logger.warn(`Status do banco indisponivel: ${this.getErrorMessage(error)}`);
    }

    return {
      status: 'error',
      timestamp: new Date().toISOString(),
      bot_ativo: false,
      arquivo_atual: null,
      progresso: 0,
      corrigidos_sessao: 0,
      ultima_correcao_sessao: null,
      ultima_atualizacao: null,
      total_registros_9ano: 0,
      total_registros_5ano: 0,
      database: 'disconnected',
      google_drive: googleDriveStatus,
    };
  }

  private getErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : 'erro desconhecido';
  }
}
