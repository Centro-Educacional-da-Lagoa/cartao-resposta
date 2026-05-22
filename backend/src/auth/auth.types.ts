export type UsuarioResponse = {
  id: string;
  nome: string;
  email: string;
  createdAt: Date;
};

export type AuthTokenPayload = {
  sub: string;
  email: string;
  iat: number;
  exp: number;
};

export type AuthResponse = {
  user: UsuarioResponse;
  accessToken: string;
  expiresIn: number;
};
