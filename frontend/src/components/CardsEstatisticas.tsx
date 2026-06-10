import { useMemo } from 'react';
import { type Aluno, type Ano } from '../service/api';
import { Users, TrendingUp, Award, AlertCircle } from 'lucide-react';

interface Props {
    ano: Ano;
    alunos: Aluno[];
}

export function CardsEstatisticas({ ano, alunos }: Props) {
    const stats = useMemo(() => {
        if (!alunos || alunos.length === 0) {
            return {
                total_alunos: 0,
                media_geral: 0,
                nota_mais_alta: 0,
                nota_mais_baixa: 0,
                aprovados: 0,
                reprovados: 0
            };
        }

        const parsePorcentagem = (valor: string): number => {
            const numero = parseFloat(valor.replace('%', '').replace(',', '.'));
            return isNaN(numero) ? 0 : numero;
        };

        const notas = alunos.map(a => parsePorcentagem(a.Porcentagem));
        const aprovados = notas.filter(n => n >= 70).length;

        return {
            total_alunos: alunos.length,
            media_geral: notas.reduce((a, b) => a + b, 0) / notas.length,
            nota_mais_alta: Math.max(...notas),
            nota_mais_baixa: Math.min(...notas),
            aprovados,
            reprovados: alunos.length - aprovados
        };
    }, [alunos]);

    if (!alunos || alunos.length === 0) return null;

    const taxaAprovacao = stats.total_alunos > 0
        ? ((stats.aprovados / stats.total_alunos) * 100).toFixed(1)
        : '0';

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-1 border-l-4 border-blue-500">
                <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-xl bg-blue-100 flex items-center justify-center flex-shrink-0">
                        <Users className="text-blue-600" size={28} />
                    </div>
                    <div className="flex-1">
                        <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Total de Alunos</h3>
                        <p className="text-3xl font-bold text-gray-800 mt-1">{stats.total_alunos}</p>
                        <span className="text-xs text-gray-400 mt-1 block">{ano}º ano</span>
                    </div>
                </div>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-1 border-l-4 border-green-500">
                <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-xl bg-green-100 flex items-center justify-center flex-shrink-0">
                        <TrendingUp className="text-green-600" size={28} />
                    </div>
                    <div className="flex-1">
                        <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Média Geral</h3>
                        <p className="text-3xl font-bold text-gray-800 mt-1">{stats.media_geral.toFixed(1)}%</p>
                        <span className="text-xs text-gray-400 mt-1 block">
                            Maior: {stats.nota_mais_alta.toFixed(1)}% | Menor: {stats.nota_mais_baixa.toFixed(1)}%
                        </span>
                    </div>
                </div>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-1 border-l-4 border-purple-500">
                <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-xl bg-purple-100 flex items-center justify-center flex-shrink-0">
                        <Award className="text-purple-600" size={28} />
                    </div>
                    <div className="flex-1">
                        <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Aprovados</h3>
                        <p className="text-3xl font-bold text-gray-800 mt-1">{stats.aprovados}</p>
                        <span className="text-xs text-gray-400 mt-1 block">Taxa de {taxaAprovacao}%</span>
                    </div>
                </div>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-1 border-l-4 border-orange-500">
                <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-xl bg-orange-100 flex items-center justify-center flex-shrink-0">
                        <AlertCircle className="text-orange-600" size={28} />
                    </div>
                    <div className="flex-1">
                        <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">Reprovados</h3>
                        <p className="text-3xl font-bold text-gray-800 mt-1">{stats.reprovados}</p>
                        <span className="text-xs text-gray-400 mt-1 block">
                            {((stats.reprovados / stats.total_alunos) * 100).toFixed(1)}% do total
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
