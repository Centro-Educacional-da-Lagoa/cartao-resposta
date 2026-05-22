import { Type } from 'class-transformer';
import {
  IsDateString,
  IsEnum,
  IsInt,
  IsNumber,
  IsString,
  Max,
  MaxLength,
  Min,
} from 'class-validator';

import { AnoEscolar } from '../ano-escolar.enum';

export class CreateResultadoAlunoDto {
  @IsDateString()
  data: string;

  @IsEnum(AnoEscolar)
  anoEscolar: AnoEscolar;

  @IsString()
  @MaxLength(255)
  escola: string;

  @IsString()
  @MaxLength(255)
  nomeCompleto: string;

  @IsDateString()
  dataNascimento: string;

  @IsString()
  @MaxLength(80)
  turma: string;

  @Type(() => Number)
  @IsInt()
  @Min(0)
  acertosLinguaPortuguesa: number;

  @Type(() => Number)
  @IsInt()
  @Min(0)
  acertosMatematica: number;

  @Type(() => Number)
  @IsInt()
  @Min(0)
  errosLinguaPortuguesa: number;

  @Type(() => Number)
  @IsInt()
  @Min(0)
  errosMatematica: number;

  @Type(() => Number)
  @IsInt()
  @Min(0)
  questoesAnuladas: number;

  @Type(() => Number)
  @IsNumber({ maxDecimalPlaces: 2 })
  @Min(0)
  @Max(100)
  porcentagemAcertos: number;
}
