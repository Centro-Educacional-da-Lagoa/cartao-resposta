import { randomUUID } from 'node:crypto';

import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma, ResultadoAluno } from '@prisma/client';

import { PrismaService } from '../prisma/prisma.service';
import { AnoEscolar } from './ano-escolar.enum';
import { AlunoLegadoResponse, ListarAlunosResponse } from './aluno.types';
import { CreateResultadoAlunoDto } from './dto/create-resultado-aluno.dto';
import { UpdateResultadoAlunoDto } from './dto/update-resultado-aluno.dto';

type ResultadoAlunoResponse = Omit<ResultadoAluno, 'porcentagemAcertos'> & {
  porcentagemAcertos: number;
};

@Injectable()
export class AlunoService {
  constructor(private readonly prisma: PrismaService) {}

  async create(dto: CreateResultadoAlunoDto): Promise<ResultadoAlunoResponse> {
    const resultado = await this.prisma.resultadoAluno.create({
      data: {
        id: randomUUID(),
        ...dto,
        data: this.toDateOnly(dto.data),
        dataNascimento: this.toDateOnly(dto.dataNascimento),
      },
    });

    return this.normalizeResultado(resultado);
  }

  async findAll(): Promise<ResultadoAlunoResponse[]> {
    const resultados = await this.prisma.resultadoAluno.findMany({
      orderBy: [{ data: 'desc' }, { nomeCompleto: 'asc' }],
    });

    return resultados.map((resultado) => this.normalizeResultado(resultado));
  }

  async findOne(id: string): Promise<ResultadoAlunoResponse> {
    const resultado = await this.prisma.resultadoAluno.findUnique({ where: { id } });

    if (!resultado) {
      throw new NotFoundException('Resultado do aluno não encontrado');
    }

    return this.normalizeResultado(resultado);
  }

  async update(id: string, dto: UpdateResultadoAlunoDto): Promise<ResultadoAlunoResponse> {
    await this.findOne(id);

    const updated = await this.prisma.resultadoAluno.update({
      where: { id },
      data: {
        ...dto,
        data: dto.data ? this.toDateOnly(dto.data) : undefined,
        dataNascimento: dto.dataNascimento ? this.toDateOnly(dto.dataNascimento) : undefined,
      },
    });

    return this.normalizeResultado(updated);
  }

  async remove(id: string): Promise<void> {
    await this.findOne(id);
    await this.prisma.resultadoAluno.delete({ where: { id } });
  }

  async listByAno(anoEscolar: AnoEscolar): Promise<ListarAlunosResponse> {
    const alunos = await this.prisma.resultadoAluno.findMany({
      where: { anoEscolar },
      orderBy: [{ data: 'desc' }, { nomeCompleto: 'asc' }],
    });

    return {
      status: 'success',
      ano: this.formatAno(anoEscolar),
      total_alunos: alunos.length,
      alunos: alunos.map((aluno) => this.toLegacyResponse(aluno)),
    };
  }

  async countByAno(anoEscolar: AnoEscolar): Promise<number> {
    return this.prisma.resultadoAluno.count({ where: { anoEscolar } });
  }

  async findLatest(
    where?: Prisma.ResultadoAlunoWhereInput,
  ): Promise<ResultadoAlunoResponse | null> {
    const resultado = await this.prisma.resultadoAluno.findFirst({
      where,
      orderBy: [{ data: 'desc' }, { updatedAt: 'desc' }],
    });

    if (!resultado) {
      return null;
    }

    return this.normalizeResultado(resultado);
  }

  toLegacyResponse(resultado: ResultadoAluno): AlunoLegadoResponse {
    return {
      id: resultado.id,
      anoEscolar: resultado.anoEscolar as AnoEscolar,
      DATA: this.formatDatePtBr(resultado.data),
      Escola: resultado.escola,
      'Nome completo': resultado.nomeCompleto,
      'Data de nascimento': this.formatDatePtBr(resultado.dataNascimento),
      Turma: resultado.turma,
      'Acertos Língua Portuguesa': String(resultado.acertosLinguaPortuguesa),
      'Acertos Matemática': String(resultado.acertosMatematica),
      'Erros Língua Portuguesa': String(resultado.errosLinguaPortuguesa),
      'Erros Matemática': String(resultado.errosMatematica),
      'Questões anuladas': String(resultado.questoesAnuladas),
      Porcentagem: `${this.toNumber(resultado.porcentagemAcertos).toFixed(2)}%`,
    };
  }

  formatAno(anoEscolar: AnoEscolar): string {
    return anoEscolar === AnoEscolar.NONO_ANO ? '9º Ano' : '5º Ano';
  }

  formatDatePtBr(value: Date | string): string {
    if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}/.test(value)) {
      const [year, month, day] = value.slice(0, 10).split('-');
      return `${day}/${month}/${year}`;
    }

    const date = value instanceof Date ? value : new Date(value);
    if (Number.isNaN(date.getTime())) {
      return '';
    }

    return new Intl.DateTimeFormat('pt-BR', { timeZone: 'UTC' }).format(date);
  }

  private toDateOnly(value: string): Date {
    return new Date(`${value.slice(0, 10)}T00:00:00.000Z`);
  }

  private normalizeResultado(resultado: ResultadoAluno): ResultadoAlunoResponse {
    return {
      ...resultado,
      porcentagemAcertos: this.toNumber(resultado.porcentagemAcertos),
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
}
