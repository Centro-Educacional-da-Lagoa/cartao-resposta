import { Controller, Get, UseGuards } from '@nestjs/common';

import { AnoEscolar } from '../aluno/ano-escolar.enum';
import { AuthGuard } from '../auth/auth.guard';
import { EstatisticasService } from './estatisticas.service';

@Controller('estatisticas')
@UseGuards(AuthGuard)
export class EstatisticasController {
  constructor(private readonly estatisticasService: EstatisticasService) {}

  @Get('9ano')
  getNonoAno() {
    return this.estatisticasService.getPorAno(AnoEscolar.NONO_ANO);
  }

  @Get('5ano')
  getQuintoAno() {
    return this.estatisticasService.getPorAno(AnoEscolar.QUINTO_ANO);
  }

  @Get('geral')
  getGeral() {
    return this.estatisticasService.getGeral();
  }
}
