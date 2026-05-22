import { Controller, Get, UseGuards } from '@nestjs/common';

import { AnoEscolar } from '../aluno/ano-escolar.enum';
import { AuthGuard } from '../auth/auth.guard';
import { GoogleDriveService } from './google-drive.service';

@Controller('pasta')
@UseGuards(AuthGuard)
export class GoogleDriveController {
  constructor(private readonly googleDriveService: GoogleDriveService) {}

  @Get('9ano')
  getPastaNonoAno() {
    return this.googleDriveService.listFilesByAno(AnoEscolar.NONO_ANO);
  }

  @Get('5ano')
  getPastaQuintoAno() {
    return this.googleDriveService.listFilesByAno(AnoEscolar.QUINTO_ANO);
  }
}
