import { AnoEscolar } from '../aluno/ano-escolar.enum';

export type VultrS3File = {
  id: string;
  name: string;
  mimeType: string;
  createdTime: string;
  size: string;
};

export type PastaResponse = {
  status: 'success';
  pasta: AnoEscolar;
  descricao: string;
  total_registros: number;
  arquivos: VultrS3File[];
};

export type UploadResponse = {
  status: 'success';
  message: string;
  filename: string;
  key: string;
};
