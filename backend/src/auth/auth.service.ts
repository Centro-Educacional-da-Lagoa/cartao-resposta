import { BadRequestException, Injectable, UnauthorizedException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { google } from 'googleapis';

import { AuthResponse } from './auth.types';
import { GoogleLoginDto } from './dto/google-login.dto';
import { LoginDto } from './dto/login.dto';
import { RegisterDto } from './dto/register.dto';
import { PasswordService } from './password.service';
import { TokenService } from './token.service';
import { UsuarioService } from './usuario.service';

@Injectable()
export class AuthService {
  private readonly googleClient = new google.auth.OAuth2();

  constructor(
    private readonly usuarioService: UsuarioService,
    private readonly passwordService: PasswordService,
    private readonly tokenService: TokenService,
    private readonly configService: ConfigService,
  ) {}

  async register(dto: RegisterDto): Promise<AuthResponse> {
    const user = await this.usuarioService.createWithPassword(dto.nome, dto.email, dto.password);
    return this.createSession(user);
  }

  async login(dto: LoginDto): Promise<AuthResponse> {
    const user = await this.usuarioService.findByEmailWithSecrets(dto.email);
    const isValidPassword = await this.passwordService.verify(dto.password, user?.senhaHash ?? null);

    if (!user || !isValidPassword) {
      throw new UnauthorizedException('E-mail ou senha inválidos');
    }

    return this.createSession(user);
  }

  async loginWithGoogle(dto: GoogleLoginDto): Promise<AuthResponse> {
    const clientId = this.configService.get<string>('GOOGLE_OAUTH_CLIENT_ID');
    if (!clientId) {
      throw new BadRequestException('Login com Google não configurado');
    }

    const ticket = await this.googleClient.verifyIdToken({
      idToken: dto.credential,
      audience: clientId,
    });
    const payload = ticket.getPayload();

    if (!payload?.sub || !payload.email || !payload.email_verified) {
      throw new UnauthorizedException('Conta Google não verificada');
    }

    const user = await this.usuarioService.findOrCreateGoogleUser({
      email: payload.email,
      nome: payload.name || payload.email.split('@')[0],
      googleId: payload.sub,
    });

    return this.createSession(user);
  }

  async getCurrentUser(id: string) {
    const user = await this.usuarioService.findById(id);
    return this.usuarioService.toResponse(user);
  }

  private createSession(user: { id: string; email: string; nome: string; createdAt: Date }): AuthResponse {
    const accessToken = this.tokenService.sign(user);

    return {
      user: this.usuarioService.toResponse(user),
      accessToken,
      expiresIn: this.tokenService.expiresInSeconds,
    };
  }
}
