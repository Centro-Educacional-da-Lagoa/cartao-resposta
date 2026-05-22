import { Module } from '@nestjs/common';
import { AlunoModule } from '../aluno/aluno.module';
import { AuthModule } from '../auth/auth.module';
import { EstatisticasController } from './estatisticas.controller';
import { EstatisticasService } from './estatisticas.service';

@Module({
  imports: [AuthModule, AlunoModule],
  controllers: [EstatisticasController],
  providers: [EstatisticasService],
})
export class EstatisticasModule {}
