import {
  Controller,
  Get,
  Post,
  UploadedFile,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { memoryStorage } from 'multer';

import { AnoEscolar } from '../aluno/ano-escolar.enum';
import { AuthGuard } from '../auth/auth.guard';
import { VultrS3Service } from './vultr-s3.service';

@Controller()
@UseGuards(AuthGuard)
export class VultrS3Controller {
  constructor(private readonly vultrS3Service: VultrS3Service) {}

  @Post('upload')
  @UseInterceptors(
    FileInterceptor('arquivo', {
      storage: memoryStorage(),
      limits: {
        files: 1,
        fileSize: 25 * 1024 * 1024,
      },
    }),
  )
  uploadPdf(@UploadedFile() file?: Express.Multer.File) {
    return this.vultrS3Service.uploadPdf(file);
  }

  @Get('pasta/4ano')
  getPastaQuartoAno() {
    return this.vultrS3Service.listFilesByAno(AnoEscolar.QUARTO_ANO);
  }

  @Get('pasta/5ano')
  getPastaQuintoAno() {
    return this.vultrS3Service.listFilesByAno(AnoEscolar.QUINTO_ANO);
  }

  @Get('pasta/8ano')
  getPastaOitavoAno() {
    return this.vultrS3Service.listFilesByAno(AnoEscolar.OITAVO_ANO);
  }

  @Get('pasta/9ano')
  getPastaNonoAno() {
    return this.vultrS3Service.listFilesByAno(AnoEscolar.NONO_ANO);
  }
}
