import { createHmac, timingSafeEqual } from 'node:crypto';

import { Injectable, InternalServerErrorException, UnauthorizedException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

import { AuthTokenPayload } from './auth.types';

const DEFAULT_EXPIRES_IN_SECONDS = 60 * 60 * 24 * 7;
const DEV_SECRET =
  'cartao-resposta-local-development-secret-change-before-production-2026';

@Injectable()
export class TokenService {
  constructor(private readonly configService: ConfigService) {}

  get expiresInSeconds(): number {
    return Number(
      this.configService.get<string>('JWT_EXPIRES_IN_SECONDS') ?? DEFAULT_EXPIRES_IN_SECONDS,
    );
  }

  sign(user: { id: string; email: string }): string {
    const now = Math.floor(Date.now() / 1000);
    const payload: AuthTokenPayload = {
      sub: user.id,
      email: user.email,
      iat: now,
      exp: now + this.expiresInSeconds,
    };

    const header = this.encodeJson({ alg: 'HS256', typ: 'JWT' });
    const body = this.encodeJson(payload);
    const signature = this.signContent(`${header}.${body}`);

    return `${header}.${body}.${signature}`;
  }

  verify(token: string): AuthTokenPayload {
    const parts = token.split('.');
    if (parts.length !== 3) {
      throw new UnauthorizedException('Sessão inválida');
    }

    const [header, body, signature] = parts;
    const expectedSignature = this.signContent(`${header}.${body}`);
    if (!this.safeCompare(signature, expectedSignature)) {
      throw new UnauthorizedException('Sessão inválida');
    }

    const payload = this.decodePayload(body);
    const now = Math.floor(Date.now() / 1000);
    if (payload.exp <= now) {
      throw new UnauthorizedException('Sessão expirada');
    }

    return payload;
  }

  private encodeJson(value: unknown): string {
    return Buffer.from(JSON.stringify(value), 'utf8').toString('base64url');
  }

  private decodePayload(value: string): AuthTokenPayload {
    try {
      const payload = JSON.parse(Buffer.from(value, 'base64url').toString('utf8'));
      if (
        typeof payload.sub !== 'string' ||
        typeof payload.email !== 'string' ||
        typeof payload.iat !== 'number' ||
        typeof payload.exp !== 'number'
      ) {
        throw new Error('invalid payload');
      }

      return payload;
    } catch {
      throw new UnauthorizedException('Sessão inválida');
    }
  }

  private signContent(content: string): string {
    return createHmac('sha256', this.getSecret()).update(content).digest('base64url');
  }

  private safeCompare(value: string, expected: string): boolean {
    const valueBuffer = Buffer.from(value);
    const expectedBuffer = Buffer.from(expected);

    return (
      valueBuffer.length === expectedBuffer.length && timingSafeEqual(valueBuffer, expectedBuffer)
    );
  }

  private getSecret(): string {
    const secret = this.configService.get<string>('JWT_SECRET');
    if (secret && secret.length >= 32) {
      return secret;
    }

    if (this.configService.get<string>('NODE_ENV') === 'production') {
      throw new InternalServerErrorException('JWT_SECRET precisa ter pelo menos 32 caracteres');
    }

    return DEV_SECRET;
  }
}
