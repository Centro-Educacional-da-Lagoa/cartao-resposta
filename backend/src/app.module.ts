import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AlunoModule } from './aluno/aluno.module';
import { AuthModule } from './auth/auth.module';
import { EstatisticasModule } from './estatisticas/estatisticas.module';
import { GoogleDriveModule } from './google-drive/google-drive.module';
import { PrismaModule } from './prisma/prisma.module';
import { StatusModule } from './status/status.module';

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
    GoogleDriveModule,
    StatusModule,
  ],
})
export class AppModule {}
