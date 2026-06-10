import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  Param,
  Patch,
  Post,
  UseGuards,
} from '@nestjs/common';

import { AuthGuard } from '../auth/auth.guard';
import { Public } from '../auth/public.decorator';
import { AnoEscolar } from './ano-escolar.enum';
import { AlunoService } from './aluno.service';
import { CreateResultadoAlunoDto } from './dto/create-resultado-aluno.dto';
import { UpdateResultadoAlunoDto } from './dto/update-resultado-aluno.dto';

@Controller('aluno')
@UseGuards(AuthGuard)
export class AlunoController {
  constructor(private readonly alunoService: AlunoService) {}

  @Get('4ano')
  listQuartoAno() {
    return this.alunoService.listByAno(AnoEscolar.QUARTO_ANO);
  }

  @Get('5ano')
  listQuintoAno() {
    return this.alunoService.listByAno(AnoEscolar.QUINTO_ANO);
  }

  @Get('8ano')
  listOitavoAno() {
    return this.alunoService.listByAno(AnoEscolar.OITAVO_ANO);
  }

  @Get('9ano')
  listNonoAno() {
    return this.alunoService.listByAno(AnoEscolar.NONO_ANO);
  }

  @Post()
  @Public()
  create(@Body() dto: CreateResultadoAlunoDto) {
    return this.alunoService.create(dto);
  }

  @Get()
  findAll() {
    return this.alunoService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.alunoService.findOne(id);
  }

  @Patch(':id')
  update(@Param('id') id: string, @Body() dto: UpdateResultadoAlunoDto) {
    return this.alunoService.update(id, dto);
  }

  @Delete(':id')
  @HttpCode(204)
  remove(@Param('id') id: string) {
    return this.alunoService.remove(id);
  }
}
