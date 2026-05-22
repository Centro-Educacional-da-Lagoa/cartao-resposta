import { randomUUID } from 'node:crypto';

import { ConflictException, Injectable, UnauthorizedException } from '@nestjs/common';
import { Prisma, Usuario } from '@prisma/client';

import { PrismaService } from '../prisma/prisma.service';
import { UsuarioResponse } from './auth.types';
import { PasswordService } from './password.service';

type UsuarioPublic = Pick<Usuario, 'id' | 'nome' | 'email' | 'createdAt'>;
type UsuarioWithSecrets = Prisma.UsuarioGetPayload<{
  select: {
    id: true;
    nome: true;
    email: true;
    createdAt: true;
    updatedAt: true;
    senhaHash: true;
    googleId: true;
  };
}>;

@Injectable()
export class UsuarioService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly passwordService: PasswordService,
  ) {}

  async createWithPassword(nome: string, email: string, password: string): Promise<UsuarioPublic> {
    const normalizedEmail = this.normalizeEmail(email);
    const existingUser = await this.findByEmailWithSecrets(normalizedEmail);
    const senhaHash = await this.passwordService.hash(password);

    if (existingUser) {
      if (existingUser.senhaHash) {
        throw new ConflictException('E-mail já cadastrado');
      }

      return this.prisma.usuario.update({
        where: { id: existingUser.id },
        data: { nome, senhaHash },
        select: this.publicSelect(),
      });
    }

    return this.prisma.usuario.create({
      data: {
        id: randomUUID(),
        nome,
        email: normalizedEmail,
        senhaHash,
        googleId: null,
      },
      select: this.publicSelect(),
    });
  }

  async findByEmailWithSecrets(email: string): Promise<UsuarioWithSecrets | null> {
    return this.prisma.usuario.findFirst({
      where: { email: this.normalizeEmail(email) },
      select: {
        ...this.publicSelect(),
        updatedAt: true,
        senhaHash: true,
        googleId: true,
      },
    });
  }

  async findById(id: string): Promise<UsuarioPublic> {
    const user = await this.prisma.usuario.findUnique({
      where: { id },
      select: this.publicSelect(),
    });
    if (!user) {
      throw new UnauthorizedException('Usuário não encontrado');
    }

    return user;
  }

  async findOrCreateGoogleUser(input: {
    email: string;
    nome: string;
    googleId: string;
  }): Promise<UsuarioPublic> {
    const normalizedEmail = this.normalizeEmail(input.email);
    const existingUser = await this.findByEmailWithSecrets(normalizedEmail);

    if (existingUser) {
      const nome = existingUser.nome || input.nome;
      return this.prisma.usuario.update({
        where: { id: existingUser.id },
        data: { nome, googleId: input.googleId },
        select: this.publicSelect(),
      });
    }

    return this.prisma.usuario.create({
      data: {
        id: randomUUID(),
        nome: input.nome,
        email: normalizedEmail,
        senhaHash: null,
        googleId: input.googleId,
      },
      select: this.publicSelect(),
    });
  }

  toResponse(user: Pick<Usuario, 'id' | 'nome' | 'email' | 'createdAt'>): UsuarioResponse {
    return {
      id: user.id,
      nome: user.nome,
      email: user.email,
      createdAt: user.createdAt,
    };
  }

  private normalizeEmail(email: string): string {
    return email.trim().toLowerCase();
  }

  private publicSelect() {
    return {
      id: true,
      nome: true,
      email: true,
      createdAt: true,
    } as const;
  }
}
