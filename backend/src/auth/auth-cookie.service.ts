import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Response } from 'express';

@Injectable()
export class AuthCookieService {
  constructor(private readonly configService: ConfigService) {}

  get cookieName(): string {
    return this.configService.get<string>('AUTH_COOKIE_NAME') ?? 'cartao_auth';
  }

  setAuthCookie(response: Response, token: string, maxAgeSeconds: number): void {
    response.setHeader('Set-Cookie', this.serializeCookie(token, maxAgeSeconds));
  }

  clearAuthCookie(response: Response): void {
    response.setHeader('Set-Cookie', this.serializeCookie('', 0));
  }

  private serializeCookie(value: string, maxAgeSeconds: number): string {
    const secure =
      this.configService.get<string>('AUTH_COOKIE_SECURE')?.toLowerCase() === 'true'
        ? 'Secure'
        : '';

    return [
      `${this.cookieName}=${value}`,
      'HttpOnly',
      'Path=/',
      `Max-Age=${maxAgeSeconds}`,
      'SameSite=Lax',
      secure,
    ]
      .filter(Boolean)
      .join('; ');
  }
}
