import {
  randomBytes,
  scrypt as scryptCallback,
  timingSafeEqual,
  type ScryptOptions,
} from 'node:crypto';

import { Injectable } from '@nestjs/common';

const SCRYPT_N = 16384;
const SCRYPT_R = 8;
const SCRYPT_P = 1;
const KEY_LENGTH = 64;
const SCRYPT_OPTIONS: ScryptOptions = {
  N: SCRYPT_N,
  r: SCRYPT_R,
  p: SCRYPT_P,
  maxmem: 64 * 1024 * 1024,
};

@Injectable()
export class PasswordService {
  async hash(password: string): Promise<string> {
    const salt = randomBytes(16).toString('base64url');
    const derivedKey = await this.deriveKey(password, salt, KEY_LENGTH, SCRYPT_OPTIONS);

    return ['scrypt', SCRYPT_N, SCRYPT_R, SCRYPT_P, salt, derivedKey.toString('base64url')].join(
      '$',
    );
  }

  async verify(password: string, storedHash: string | null): Promise<boolean> {
    if (!storedHash) {
      return false;
    }

    const [algorithm, n, r, p, salt, hash] = storedHash.split('$');
    if (algorithm !== 'scrypt' || !n || !r || !p || !salt || !hash) {
      return false;
    }

    const expectedHash = Buffer.from(hash, 'base64url');
    const candidateHash = await this.deriveKey(password, salt, expectedHash.length, {
      N: Number(n),
      r: Number(r),
      p: Number(p),
      maxmem: 64 * 1024 * 1024,
    });

    return (
      candidateHash.length === expectedHash.length && timingSafeEqual(candidateHash, expectedHash)
    );
  }

  private deriveKey(
    password: string,
    salt: string,
    keyLength: number,
    options: ScryptOptions,
  ): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      scryptCallback(password, salt, keyLength, options, (error, derivedKey) => {
        if (error) {
          reject(error);
          return;
        }

        resolve(derivedKey);
      });
    });
  }
}
