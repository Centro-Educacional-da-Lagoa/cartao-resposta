import { Injectable, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { drive_v3, google } from 'googleapis';

import { AnoEscolar } from '../aluno/ano-escolar.enum';
import { GoogleDriveFile, PastaResponse } from './google-drive.types';

@Injectable()
export class GoogleDriveService {
  private driveClient?: drive_v3.Drive;

  constructor(private readonly configService: ConfigService) {}

  isConfigured(anoEscolar?: AnoEscolar): boolean {
    const hasCredentials = Boolean(this.configService.get<string>('GOOGLE_CREDENTIALS_JSON'));

    if (!anoEscolar) {
      return hasCredentials;
    }

    return hasCredentials && Boolean(this.getFolderId(anoEscolar));
  }

  async listFilesByAno(anoEscolar: AnoEscolar): Promise<PastaResponse> {
    const folderId = this.getFolderId(anoEscolar);
    if (!folderId) {
      throw new ServiceUnavailableException('Pasta do Google Drive não configurada');
    }

    const drive = this.getDriveClient();
    const results = await drive.files.list({
      q: `'${folderId}' in parents and trashed = false`,
      fields: 'files(id, name, mimeType, createdTime, size)',
      orderBy: 'createdTime desc',
      pageSize: 1000,
      supportsAllDrives: true,
      includeItemsFromAllDrives: true,
    });

    const arquivos = (results.data.files ?? []) as GoogleDriveFile[];

    return {
      status: 'success',
      pasta: anoEscolar,
      descricao: `Pasta de dados do ${this.formatAno(anoEscolar)}`,
      total_registros: arquivos.length,
      arquivos,
    };
  }

  private getDriveClient(): drive_v3.Drive {
    if (this.driveClient) {
      return this.driveClient;
    }

    const credentialsJson = this.configService.get<string>('GOOGLE_CREDENTIALS_JSON');
    if (!credentialsJson) {
      throw new ServiceUnavailableException('Credenciais do Google Drive não configuradas');
    }

    const credentials = this.parseCredentials(credentialsJson);
    const auth = new google.auth.GoogleAuth({
      credentials,
      scopes: ['https://www.googleapis.com/auth/drive.metadata.readonly'],
    });

    this.driveClient = google.drive({ version: 'v3', auth });
    return this.driveClient;
  }

  private parseCredentials(credentialsJson: string) {
    try {
      const credentials = JSON.parse(credentialsJson);

      if (typeof credentials.private_key === 'string') {
        credentials.private_key = credentials.private_key.replace(/\\n/g, '\n');
      }

      return credentials;
    } catch {
      throw new ServiceUnavailableException('Credenciais do Google Drive inválidas');
    }
  }

  private getFolderId(anoEscolar: AnoEscolar): string | undefined {
    const numeroAno = anoEscolar.replace('ano', '');
    return (
      this.configService.get<string>(`DRIVE_FOLDER_${numeroAno}ANO`) ??
      this.configService.get<string>(`DRIVER_FOLDER_${numeroAno}ANO`)
    );
  }

  private formatAno(anoEscolar: AnoEscolar): string {
    return anoEscolar.replace('ano', 'o ano');
  }
}
