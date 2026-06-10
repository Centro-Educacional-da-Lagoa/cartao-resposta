import { Controller, Get, UseGuards } from '@nestjs/common';

import { AnoEscolar } from '../aluno/ano-escolar.enum';
import { AuthGuard } from '../auth/auth.guard';
import { EstatisticasService } from './estatisticas.service';

@Controller('estatisticas')
@UseGuards(AuthGuard)
export class EstatisticasController {
  constructor(private readonly estatisticasService: EstatisticasService) {}

  @Get('4ano')
  getQuartoAno() {
    return this.estatisticasService.getPorAno(AnoEscolar.QUARTO_ANO);
  }

  @Get('5ano')
  getQuintoAno() {
    return this.estatisticasService.getPorAno(AnoEscolar.QUINTO_ANO);
  }

  @Get('8ano')
  getOitavoAno() {
    return this.estatisticasService.getPorAno(AnoEscolar.OITAVO_ANO);
  }

  @Get('9ano')
  getNonoAno() {
    return this.estatisticasService.getPorAno(AnoEscolar.NONO_ANO);
  }

  @Get('geral')
  getGeral() {
    return this.estatisticasService.getGeral();
  }
}
