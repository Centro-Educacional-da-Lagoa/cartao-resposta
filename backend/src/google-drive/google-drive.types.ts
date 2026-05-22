export type GoogleDriveFile = {
  id?: string | null;
  name?: string | null;
  mimeType?: string | null;
  createdTime?: string | null;
  size?: string | null;
};

export type PastaResponse = {
  status: 'success';
  pasta: '5ano' | '9ano';
  descricao: string;
  total_registros: number;
  arquivos: GoogleDriveFile[];
};
