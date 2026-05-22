import { AnoEscolar } from '../aluno/ano-escolar.enum';

export type EstatisticasResponse = {
  status: 'success';
  ano: string;
  total_alunos: number;
  media_geral: number;
  nota_mais_alta: number;
  nota_mais_baixa: number;
  aprovados: number;
  reprovados: number;
};

export type EstatisticasGeralResponse = {
  status: 'success';
  total_alunos: number;
  media_geral: number;
  nota_mais_alta: number;
  nota_mais_baixa: number;
  aprovados: number;
  reprovados: number;
  por_ano: Record<AnoEscolar, Omit<EstatisticasResponse, 'status' | 'ano'>>;
};
