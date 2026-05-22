import { Type } from 'class-transformer';
import {
  IsDateString,
  IsEnum,
  IsInt,
  IsNumber,
  IsOptional,
  IsString,
  Max,
  MaxLength,
  Min,
} from 'class-validator';

import { AnoEscolar } from '../ano-escolar.enum';

export class UpdateResultadoAlunoDto {
  @IsOptional()
  @IsDateString()
  data?: string;

  @IsOptional()
  @IsEnum(AnoEscolar)
  anoEscolar?: AnoEscolar;

  @IsOptional()
  @IsString()
  @MaxLength(255)
  escola?: string;

  @IsOptional()
  @IsString()
  @MaxLength(255)
  nomeCompleto?: string;

  @IsOptional()
  @IsDateString()
  dataNascimento?: string;

  @IsOptional()
  @IsString()
  @MaxLength(80)
  turma?: string;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(0)
  acertosLinguaPortuguesa?: number;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(0)
  acertosMatematica?: number;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(0)
  errosLinguaPortuguesa?: number;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(0)
  errosMatematica?: number;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(0)
  questoesAnuladas?: number;

  @IsOptional()
  @Type(() => Number)
  @IsNumber({ maxDecimalPlaces: 2 })
  @Min(0)
  @Max(100)
  porcentagemAcertos?: number;
}
