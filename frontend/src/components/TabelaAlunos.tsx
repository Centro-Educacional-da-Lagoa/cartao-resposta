import { type Aluno, type AnoFiltro } from '../service/api';
import { BotoesExportacao } from './BotoesExportacao';
import { useMemo, useState } from 'react';
import { Search } from 'lucide-react';

interface Props {
    alunos: Aluno[];
    ano: AnoFiltro;
}

interface FiltrosData {
    busca: string;
    escola: string;
    turma: string;
}

export function TabelaAlunos({ alunos, ano }: Props) {
    const [filtros, setFiltros] = useState<FiltrosData>({
        busca: '',
        escola: '',
        turma: ''
    });
    function parsePorcentagem(valor: string): number {
        if (!valor) return 0;
        const numero = parseFloat(valor.replace('%', '').replace(',', '.'));
        return isNaN(numero) ? 0 : numero;
    }

    const filtroAluno = useMemo(() => {
        return alunos.filter(aluno => {
            if (filtros.busca && !aluno["Nome completo"].toLowerCase().includes(filtros.busca.toLowerCase())) {
                return false;
            }
            if (filtros.escola && aluno.Escola !== filtros.escola) {
                return false;
            }
            return true;
        })
    }, [alunos, filtros]);

    const AlunosOrdenados = useMemo(() => {
        return [...filtroAluno].sort((a, b) => {
            const dataA = new Date(a.DATA.split('/').reverse().join('-'));
            const dataB = new Date(b.DATA.split('/').reverse().join('-'));
            return dataB.getTime() - dataA.getTime();
        });
    }, [filtroAluno]);


    const escolasUnicas = useMemo(() =>
        [...new Set(alunos.map(a => a.Escola).filter(Boolean))].sort(),
        [alunos]
    );

    const handleChange = (campo: keyof FiltrosData, valor: string) => {
        console.log(`Filtro ${campo}:`, valor);
        setFiltros(prev => ({ ...prev, [campo]: valor }));
    };



    return (
        <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 flex items-start justify-between">
                <div>
                    <h2 className="text-lg font-semibold text-gray-800">
                        📝 Lista de Alunos {ano !== 'geral' && `- ${ano}º Ano`}
                    </h2>
                    <p className="text-sm text-gray-500 mt-1">
                        {AlunosOrdenados.length} {AlunosOrdenados.length === 1 ? 'aluno encontrado' : 'alunos encontrados'}
                    </p>
                </div>
                <BotoesExportacao alunos={AlunosOrdenados} ano={ano} />
            </div>

            {/* Filtros */}
            <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                <div className="flex gap-3 items-center flex-wrap">
                    <div className="flex-1 min-w-[300px] flex items-center gap-3 bg-white border border-gray-200 rounded-lg px-4 py-2.5 transition-all focus-within:border-blue-500 focus-within:shadow-sm">
                        <Search size={18} className="text-gray-400" />
                        <input
                            type="text"
                            placeholder="Buscar por nome do aluno..."
                            value={filtros.busca}
                            onChange={(e) => handleChange('busca', e.target.value)}
                            className="flex-1 bg-transparent border-none outline-none text-gray-700 placeholder-gray-400 text-sm"
                        />
                    </div>

                    <div className="flex flex-col gap-3 mb-5">
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Escola</label>
                        <select
                            value={filtros.escola}
                            onChange={(e) => handleChange('escola', e.target.value)}
                            className="px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-700 bg-white focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100 transition-all"
                        >
                            <option value="">Todas as escolas</option>
                            {escolasUnicas.map(escola => (
                                <option key={escola} value={escola}>{escola}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            <div className="overflow-x-auto shadow-xl border border-gray-200 rounded-lg">
                <table className="w-full">
                    <thead className="bg-gray-50 border-b border-gray-200">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Data</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Nome</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Escola</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Turma</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Nota</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Port.</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Mat.</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Erros Port.</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Erros Mat.</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Anuladas</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {AlunosOrdenados.length > 0 ? (
                            AlunosOrdenados.map((aluno, index) => (
                                <tr key={index} className="hover:bg-gray-50 transition-colors">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                                        {aluno.DATA}
                                    </td>
                                    <td className="px-6 py-4 text-sm font-medium text-gray-900">
                                        {aluno["Nome completo"]}
                                    </td>
                                    <td className="px-6 py-4 text-sm text-gray-600">
                                        {aluno.Escola}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                                        {aluno.Turma}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
                                        <span className={parsePorcentagem(aluno.Porcentagem) >= 70 ? 'text-green-600' : 'text-red-600'}>
                                            {aluno.Porcentagem}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                                        {aluno["Acertos Língua Portuguesa"]}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                                        {aluno["Acertos Matemática"]}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                                        {aluno["Erros Língua Portuguesa"]}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                                        {aluno["Erros Matemática"]}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                                        {parseInt(aluno["Questões anuladas"]) > 0 ? aluno["Questões anuladas"] : '--'}
                                    </td>
                                </tr>
                            ))
                        ) : (
                            <tr>
                                <td colSpan={10} className="px-6 py-12 text-center text-gray-500">
                                    Nenhum aluno encontrado
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
