import { AnoEscolar } from '../aluno/ano-escolar.enum';

export type GoogleDriveFile = {
  id?: string | null;
  name?: string | null;
  mimeType?: string | null;
  createdTime?: string | null;
  size?: string | null;
};

export type PastaResponse = {
  status: 'success';
  pasta: AnoEscolar;
  descricao: string;
  total_registros: number;
  arquivos: GoogleDriveFile[];
};
