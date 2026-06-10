import { BarChart, Bar, PieChart, Pie, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { type Aluno, type Ano } from '../service/api';

interface Props {
    ano: Ano;
    alunos: Aluno[];
}

export function Graficos({ alunos }: Props) {
    if (!alunos || alunos.length === 0) {
        return (
            <div className="text-center py-12 text-gray-600">
                📊 Nenhum dado disponível para exibir gráficos.
            </div>
        );
    }

    const parsePorcentagem = (valor: string): number => {
        const numero = parseFloat(valor.replace('%', '').replace(',', '.'));
        return isNaN(numero) ? 0 : numero;
    };

    const aprovados = alunos.filter(a => parsePorcentagem(a.Porcentagem) >= 70).length;
    const reprovados = alunos.length - aprovados;
    const dadosPizza = [
        { name: 'Aprovados', value: aprovados, color: '#10b981' },
        { name: 'Reprovados', value: reprovados, color: '#ef4444' }
    ];

    const alunosPorTurma = alunos.reduce((acc, aluno) => {
        const turma = aluno.Turma || 'Sem turma';
        if (!acc[turma]) {
            acc[turma] = { turma, total: 0, mediaNotas: 0, somaNotas: 0 };
        }
        acc[turma].total += 1;
        acc[turma].somaNotas += parsePorcentagem(aluno.Porcentagem);
        return acc;
    }, {} as Record<string, { turma: string; total: number; mediaNotas: number; somaNotas: number }>);

    const dadosBarras = Object.values(alunosPorTurma).map(item => ({
        turma: item.turma,
        alunos: item.total,
        media: parseFloat((item.somaNotas / item.total).toFixed(1))
    }));

    const dadosMateria = alunos.map(aluno => ({
        nome: aluno["Nome completo"].split(' ')[0],
        portugues: parseInt(aluno["Acertos Língua Portuguesa"] || '0'),
        matematica: parseInt(aluno["Acertos Matemática"] || '0')
    })).slice(0, 10);

    const dadosPorData = alunos.reduce((acc, aluno) => {
        const data = aluno.DATA || 'Sem data';
        if (!acc[data]) {
            acc[data] = { data, notas: [] };
        }
        acc[data].notas.push(parsePorcentagem(aluno.Porcentagem));
        return acc;
    }, {} as Record<string, { data: string; notas: number[] }>);

    const dadosLinha = Object.values(dadosPorData)
        .map(item => ({
            data: item.data,
            media: parseFloat((item.notas.reduce((a, b) => a + b, 0) / item.notas.length).toFixed(1))
        }))
        .sort((a, b) => new Date(a.data).getTime() - new Date(b.data).getTime());

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Gráfico de Pizza */}
            <div className="bg-white rounded-xl p-6 shadow-sm">
                <h3 className="text-base font-semibold text-gray-800 mb-4">📊 Aprovação vs Reprovação</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                        <Pie
                            data={dadosPizza}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name}: ${((percent || 0) * 100).toFixed(0)}%`}
                            outerRadius={100}
                            fill="#8884d8"
                            dataKey="value"
                        >
                            {dadosPizza.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Pie>
                        <Tooltip />
                    </PieChart>
                </ResponsiveContainer>
            </div>

            {/* Gráfico de Barras - Turmas */}
            <div className="bg-white rounded-xl p-6 shadow-sm">
                <h3 className="text-base font-semibold text-gray-800 mb-4">📚 Desempenho por Turma</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={dadosBarras}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="turma" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="alunos" fill="#3b82f6" name="Qtd. Alunos" />
                        <Bar dataKey="media" fill="#10b981" name="Média %" />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Gráfico de Linha - Evolução */}
            <div className="bg-white rounded-xl p-6 shadow-sm lg:col-span-2">
                <h3 className="text-base font-semibold text-gray-800 mb-4">📈 Evolução Temporal das Notas</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={dadosLinha}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="data" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Legend />
                        <Line
                            type="monotone"
                            dataKey="media"
                            stroke="#8b5cf6"
                            strokeWidth={3}
                            name="Média %"
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Gráfico de Barras - Matérias */}
            <div className="bg-white rounded-xl p-6 shadow-sm lg:col-span-2">
                <h3 className="text-base font-semibold text-gray-800 mb-4">🎯 Acertos por Matéria (Top 10 Alunos)</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={dadosMateria}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="nome" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="portugues" fill="#06b6d4" name="Português" />
                        <Bar dataKey="matematica" fill="#f59e0b" name="Matemática" />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
