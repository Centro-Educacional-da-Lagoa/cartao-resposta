import { Module } from '@nestjs/common';

import { AuthModule } from '../auth/auth.module';
import { VultrS3Controller } from './vultr-s3.controller';
import { VultrS3Service } from './vultr-s3.service';

@Module({
  imports: [AuthModule],
  controllers: [VultrS3Controller],
  providers: [VultrS3Service],
  exports: [VultrS3Service],
})
export class VultrS3Module {}
