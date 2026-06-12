import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AlunoModule } from './aluno/aluno.module';
import { AuthModule } from './auth/auth.module';
import { EstatisticasModule } from './estatisticas/estatisticas.module';
import { PrismaModule } from './prisma/prisma.module';
import { StatusModule } from './status/status.module';
import { VultrS3Module } from './vultr-s3/vultr-s3.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: ['.env', 'backend/.env', '../.env'],
    }),
    PrismaModule,
    AuthModule,
    AlunoModule,
    EstatisticasModule,
    VultrS3Module,
    StatusModule,
  ],
})
export class AppModule {}
