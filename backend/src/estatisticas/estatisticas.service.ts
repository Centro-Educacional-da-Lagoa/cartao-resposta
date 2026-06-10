import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';

import { AnoEscolar } from '../aluno/ano-escolar.enum';
import { AlunoService } from '../aluno/aluno.service';
import { PrismaService } from '../prisma/prisma.service';
import { EstatisticasGeralResponse, EstatisticasResponse } from './estatisticas.types';

@Injectable()
export class EstatisticasService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly alunoService: AlunoService,
  ) {}

  async getPorAno(anoEscolar: AnoEscolar): Promise<EstatisticasResponse> {
    const estatisticas = await this.calcular(anoEscolar);

    return {
      status: 'success',
      ano: this.alunoService.formatAno(anoEscolar),
      ...estatisticas,
    };
  }

  async getGeral(): Promise<EstatisticasGeralResponse> {
    const [geral, quartoAno, quintoAno, oitavoAno, nonoAno] = await Promise.all([
      this.calcular(),
      this.calcular(AnoEscolar.QUARTO_ANO),
      this.calcular(AnoEscolar.QUINTO_ANO),
      this.calcular(AnoEscolar.OITAVO_ANO),
      this.calcular(AnoEscolar.NONO_ANO),
    ]);

    return {
      status: 'success',
      ...geral,
      por_ano: {
        [AnoEscolar.QUARTO_ANO]: quartoAno,
        [AnoEscolar.QUINTO_ANO]: quintoAno,
        [AnoEscolar.OITAVO_ANO]: oitavoAno,
        [AnoEscolar.NONO_ANO]: nonoAno,
      },
    };
  }

  private async calcular(anoEscolar?: AnoEscolar) {
    const where: Prisma.ResultadoAlunoWhereInput | undefined = anoEscolar
      ? { anoEscolar }
      : undefined;

    const [totalAlunos, agregados, aprovados] = await Promise.all([
      this.prisma.resultadoAluno.count({ where }),
      this.prisma.resultadoAluno.aggregate({
        where,
        _avg: { porcentagemAcertos: true },
        _max: { porcentagemAcertos: true },
        _min: { porcentagemAcertos: true },
      }),
      this.prisma.resultadoAluno.count({
        where: {
          ...(where ?? {}),
          porcentagemAcertos: { gte: 70 },
        },
      }),
    ]);

    return {
      total_alunos: totalAlunos,
      media_geral: this.toRoundedNumber(agregados._avg.porcentagemAcertos),
      nota_mais_alta: this.toRoundedNumber(agregados._max.porcentagemAcertos),
      nota_mais_baixa: this.toRoundedNumber(agregados._min.porcentagemAcertos),
      aprovados,
      reprovados: Math.max(totalAlunos - aprovados, 0),
    };
  }

  private toNumber(value: Prisma.Decimal | number | null | undefined): number {
    if (value === null || value === undefined) {
      return 0;
    }

    if (typeof value === 'number') {
      return value;
    }

    return Number(value.toString());
  }

  private toRoundedNumber(value: Prisma.Decimal | number | null | undefined): number {
    return Number(this.toNumber(value).toFixed(2));
  }
}
