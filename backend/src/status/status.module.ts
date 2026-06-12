import { Module } from '@nestjs/common';

import { AlunoModule } from '../aluno/aluno.module';
import { VultrS3Module } from '../vultr-s3/vultr-s3.module';
import { StatusController } from './status.controller';
import { StatusService } from './status.service';

@Module({
  imports: [AlunoModule, VultrS3Module],
  controllers: [StatusController],
  providers: [StatusService],
})
export class StatusModule {}
