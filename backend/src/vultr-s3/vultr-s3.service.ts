import {
  BadRequestException,
  Injectable,
  Logger,
  PayloadTooLargeException,
  ServiceUnavailableException,
} from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
  S3ClientConfig,
} from '@aws-sdk/client-s3';
import { randomUUID } from 'node:crypto';
import { extname, posix } from 'node:path';

import { AnoEscolar } from '../aluno/ano-escolar.enum';
import { PastaResponse, UploadResponse, VultrS3File } from './vultr-s3.types';

const PDF_MIME_TYPE = 'application/pdf';
const DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

@Injectable()
export class VultrS3Service {
  private readonly logger = new Logger(VultrS3Service.name);
  private s3Client?: S3Client;

  constructor(private readonly configService: ConfigService) {}

  isConfigured(): boolean {
    return Boolean(
      this.configService.get<string>('VULTR_S3_ACCESS_KEY_ID') &&
        this.configService.get<string>('VULTR_S3_SECRET_ACCESS_KEY') &&
        this.configService.get<string>('VULTR_S3_HOST') &&
        this.configService.get<string>('VULTR_S3_BUCKET'),
    );
  }

  getMaxUploadBytes(): number {
    return DEFAULT_MAX_UPLOAD_BYTES;
  }

  async uploadPdf(file?: Express.Multer.File): Promise<UploadResponse> {
    if (!file) {
      throw new BadRequestException('Selecione um arquivo PDF');
    }

    if (file.size > this.getMaxUploadBytes()) {
      throw new PayloadTooLargeException('O arquivo PDF excede o limite permitido');
    }

    if (!this.isPdf(file)) {
      throw new BadRequestException('Somente arquivos PDF válidos são permitidos');
    }

    const filename = this.sanitizeFilename(file.originalname);
    const key = posix.join(
      this.getUploadPrefix(),
      randomUUID(),
      filename,
    );

    try {
      await this.getClient().send(
        new PutObjectCommand({
          Bucket: this.getBucket(),
          Key: key,
          Body: file.buffer,
          ContentLength: file.size,
          ContentType: PDF_MIME_TYPE,
        }),
      );
    } catch (error) {
      this.logger.error(`Falha ao enviar PDF para o Vultr S3: ${this.getErrorMessage(error)}`);
      throw new ServiceUnavailableException('Não foi possível armazenar o PDF agora');
    }

    return {
      status: 'success',
      message: `Arquivo ${filename} enviado para processamento.`,
      filename,
      key,
    };
  }

  async listFilesByAno(anoEscolar: AnoEscolar): Promise<PastaResponse> {
    const prefix = posix.join(this.getProcessedPrefix(), anoEscolar, '/');

    try {
      const arquivos: VultrS3File[] = [];
      let continuationToken: string | undefined;

      do {
        const response = await this.getClient().send(
          new ListObjectsV2Command({
            Bucket: this.getBucket(),
            Prefix: prefix,
            ContinuationToken: continuationToken,
          }),
        );

        for (const object of response.Contents ?? []) {
          if (!object.Key || object.Key.endsWith('/')) {
            continue;
          }

          arquivos.push({
            id: object.Key,
            name: posix.basename(object.Key),
            mimeType: this.getMimeType(object.Key),
            createdTime: object.LastModified?.toISOString() ?? '',
            size: String(object.Size ?? 0),
          });
        }

        continuationToken = response.IsTruncated
          ? response.NextContinuationToken
          : undefined;
      } while (continuationToken);

      arquivos.sort((left, right) => right.createdTime.localeCompare(left.createdTime));

      return {
        status: 'success',
        pasta: anoEscolar,
        descricao: `Arquivos processados do ${this.formatAno(anoEscolar)}`,
        total_registros: arquivos.length,
        arquivos,
      };
    } catch (error) {
      this.logger.error(`Falha ao listar objetos do Vultr S3: ${this.getErrorMessage(error)}`);
      throw new ServiceUnavailableException('Não foi possível consultar os arquivos no Vultr S3');
    }
  }

  private getClient(): S3Client {
    if (this.s3Client) {
      return this.s3Client;
    }

    const accessKeyId = this.getRequiredConfig('VULTR_S3_ACCESS_KEY_ID');
    const secretAccessKey = this.getRequiredConfig('VULTR_S3_SECRET_ACCESS_KEY');
    const endpoint = this.normalizeEndpoint(this.getRequiredConfig('VULTR_S3_HOST'));
    const region =
      this.configService.get<string>('VULTR_S3_REGION')?.trim() ||
      new URL(endpoint).hostname.split('.')[0] ||
      'us-east-1';

    const config: S3ClientConfig = {
      endpoint,
      region,
      credentials: {
        accessKeyId,
        secretAccessKey,
      },
      forcePathStyle: this.getBooleanConfig('VULTR_S3_FORCE_PATH_STYLE', false),
    };

    this.s3Client = new S3Client(config);
    return this.s3Client;
  }

  private getBucket(): string {
    return this.getRequiredConfig('VULTR_S3_BUCKET');
  }

  private getUploadPrefix(): string {
    return this.normalizePrefix(
      this.configService.get<string>('VULTR_S3_PREFIX_UPLOAD') || 'entrada',
    );
  }

  private getProcessedPrefix(): string {
    return this.normalizePrefix(
      this.configService.get<string>('VULTR_S3_PREFIX_PROCESSADOS') || 'processados',
    );
  }

  private getRequiredConfig(name: string): string {
    const value = this.configService.get<string>(name)?.trim();
    if (!value) {
      throw new ServiceUnavailableException(`Configuração ${name} não encontrada`);
    }
    return value;
  }

  private normalizeEndpoint(host: string): string {
    return /^https?:\/\//i.test(host) ? host : `https://${host}`;
  }

  private normalizePrefix(prefix: string): string {
    const normalized = prefix.trim().replace(/^\/+|\/+$/g, '');
    return normalized ? `${normalized}/` : '';
  }

  private sanitizeFilename(originalName: string): string {
    const baseName = posix.basename(originalName.replace(/\\/g, '/'));
    const withoutExtension = baseName.slice(0, -extname(baseName).length);
    const safeBaseName =
      withoutExtension
        .normalize('NFKD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/[^a-zA-Z0-9._-]+/g, '_')
        .replace(/^[-_.]+|[-_.]+$/g, '')
        .slice(0, 160) || 'cartao-resposta';

    return `${safeBaseName}.pdf`;
  }

  private isPdf(file: Express.Multer.File): boolean {
    const hasPdfExtension = extname(file.originalname).toLowerCase() === '.pdf';
    const hasPdfMime = file.mimetype === PDF_MIME_TYPE;
    const hasPdfSignature = file.buffer.subarray(0, 5).toString('ascii') === '%PDF-';
    return hasPdfExtension && hasPdfMime && hasPdfSignature;
  }

  private getMimeType(key: string): string {
    return key.toLowerCase().endsWith('.pdf') ? PDF_MIME_TYPE : 'application/octet-stream';
  }

  private getBooleanConfig(name: string, fallback: boolean): boolean {
    const value = this.configService.get<string>(name);
    if (value === undefined) {
      return fallback;
    }
    return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase());
  }

  private formatAno(anoEscolar: AnoEscolar): string {
    return anoEscolar.replace('ano', 'o ano');
  }

  private getErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : 'erro desconhecido';
  }
}
