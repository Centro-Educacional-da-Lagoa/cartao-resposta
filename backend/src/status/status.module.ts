import { Module } from '@nestjs/common';

import { AlunoModule } from '../aluno/aluno.module';
import { GoogleDriveModule } from '../google-drive/google-drive.module';
import { StatusController } from './status.controller';
import { StatusService } from './status.service';

@Module({
  imports: [AlunoModule, GoogleDriveModule],
  controllers: [StatusController],
  providers: [StatusService],
})
export class StatusModule {}
