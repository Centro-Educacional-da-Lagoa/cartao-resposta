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
      folder_4ano: this.googleDriveService.isConfigured(AnoEscolar.QUARTO_ANO),
      folder_5ano: this.googleDriveService.isConfigured(AnoEscolar.QUINTO_ANO),
      folder_8ano: this.googleDriveService.isConfigured(AnoEscolar.OITAVO_ANO),
      folder_9ano: this.googleDriveService.isConfigured(AnoEscolar.NONO_ANO),
    };

    try {
      const [totalQuartoAno, totalQuintoAno, totalOitavoAno, totalNonoAno, latest] =
        await Promise.all([
          this.alunoService.countByAno(AnoEscolar.QUARTO_ANO),
          this.alunoService.countByAno(AnoEscolar.QUINTO_ANO),
          this.alunoService.countByAno(AnoEscolar.OITAVO_ANO),
          this.alunoService.countByAno(AnoEscolar.NONO_ANO),
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
        total_registros_4ano: totalQuartoAno,
        total_registros_5ano: totalQuintoAno,
        total_registros_8ano: totalOitavoAno,
        total_registros_9ano: totalNonoAno,
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
      total_registros_4ano: 0,
      total_registros_5ano: 0,
      total_registros_8ano: 0,
      total_registros_9ano: 0,
      database: 'disconnected',
      google_drive: googleDriveStatus,
    };
  }

  private getErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : 'erro desconhecido';
  }
}
