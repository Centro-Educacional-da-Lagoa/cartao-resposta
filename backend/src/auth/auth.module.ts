import { Module } from '@nestjs/common';
import { AuthCookieService } from './auth-cookie.service';
import { AuthController } from './auth.controller';
import { AuthGuard } from './auth.guard';
import { AuthService } from './auth.service';
import { PasswordService } from './password.service';
import { TokenService } from './token.service';
import { UsuarioService } from './usuario.service';

@Module({
  controllers: [AuthController],
  providers: [
    AuthService,
    AuthGuard,
    AuthCookieService,
    PasswordService,
    TokenService,
    UsuarioService,
  ],
  exports: [AuthGuard, AuthCookieService, TokenService],
})
export class AuthModule {}
