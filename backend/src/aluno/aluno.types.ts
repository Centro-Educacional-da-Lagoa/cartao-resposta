import { AnoEscolar } from './ano-escolar.enum';

export type AlunoLegadoResponse = {
  id: string;
  anoEscolar: AnoEscolar;
  DATA: string;
  Escola: string;
  'Nome completo': string;
  'Data de nascimento': string;
  Turma: string;
  'Acertos Língua Portuguesa': string;
  'Acertos Matemática': string;
  'Erros Língua Portuguesa': string;
  'Erros Matemática': string;
  'Questões anuladas': string;
  Porcentagem: string;
};

export type ListarAlunosResponse = {
  status: 'success';
  ano: string;
  total_alunos: number;
  alunos: AlunoLegadoResponse[];
};
