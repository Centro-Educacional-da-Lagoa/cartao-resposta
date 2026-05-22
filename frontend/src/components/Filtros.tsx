import { useState } from 'react';
import { Search, Filter, X } from 'lucide-react';

interface FiltrosData {
    busca: string;
    escola: string;
    turma: string;
    dataInicio: string;
    dataFim: string;
}

interface Props {
    onFiltrar: (filtros: FiltrosData) => void;
    escolas: string[];
    turmas: string[];
}

export function Filtros({ onFiltrar, escolas, turmas }: Props) {
    const [mostrarFiltros, setMostrarFiltros] = useState(false);
    const [filtros, setFiltros] = useState<FiltrosData>({
        busca: '',
        escola: '',
        turma: '',
        dataInicio: '',
        dataFim: ''
    });

    const handleChange = (campo: keyof FiltrosData, valor: string | number) => {
        const novosFiltros = { ...filtros, [campo]: valor };
        setFiltros(novosFiltros);
        onFiltrar(novosFiltros);
    };

    const limparFiltros = () => {
        const filtrosLimpos = {
            busca: '',
            escola: '',
            turma: '',
            dataInicio: '',
            dataFim: ''
        };
        setFiltros(filtrosLimpos);
        onFiltrar(filtrosLimpos);
    };

    const filtrosAtivos = Object.values(filtros).some(v =>
        typeof v === 'string' ? v !== '' : v > 0
    );

    const countFiltrosAtivos = Object.values(filtros).filter(v =>
        typeof v === 'string' ? v !== '' : v > 0
    ).length;

    return (
        <div className="bg-white rounded-xl p-5 shadow-sm mb-8">
            <div className="flex gap-3 items-center flex-wrap">
                {/* Busca Rápida */}
                <div className="flex-1 min-w-[300px] flex items-center gap-3 bg-gray-50 border border-gray-200 rounded-lg px-4 py-3 transition-all focus-within:border-blue-500 focus-within:bg-white focus-within:shadow-sm">
                    <Search size={20} className="text-gray-400" />
                    <input
                        type="text"
                        placeholder="Buscar por nome do aluno..."
                        value={filtros.busca}
                        onChange={(e) => handleChange('busca', e.target.value)}
                        className="flex-1 bg-transparent border-none outline-none text-gray-700 placeholder-gray-400"
                    />
                </div>

                {/* Botão Filtros Avançados */}
                <button
                    className={`flex items-center gap-2 px-5 py-3 rounded-lg font-medium text-sm transition-all relative ${mostrarFiltros
                            ? 'bg-blue-500 text-white border-blue-500'
                            : 'bg-gray-50 text-gray-700 border border-gray-200 hover:bg-gray-100'
                        }`}
                    onClick={() => setMostrarFiltros(!mostrarFiltros)}
                >
                    <Filter size={20} />
                    Filtros Avançados
                    {filtrosAtivos && (
                        <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs font-bold px-2 py-0.5 rounded-full min-w-[20px] text-center">
                            {countFiltrosAtivos}
                        </span>
                    )}
                </button>

                {/* Botão Limpar */}
                {filtrosAtivos && (
                    <button
                        className="flex items-center gap-2 px-4 py-3 bg-red-50 text-red-600 border border-red-200 rounded-lg font-medium text-sm hover:bg-red-100 transition-all"
                        onClick={limparFiltros}
                    >
                        <X size={20} />
                        Limpar
                    </button>
                )}
            </div>

            {/* Filtros Avançados */}
            {mostrarFiltros && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mt-5 pt-5 border-t border-gray-200">
                    <div className="flex flex-col gap-2">
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Escola</label>
                        <select
                            value={filtros.escola}
                            onChange={(e) => handleChange('escola', e.target.value)}
                            className="px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-700 bg-white focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100 transition-all"
                        >
                            <option value="">Todas as escolas</option>
                            {escolas.map(escola => (
                                <option key={escola} value={escola}>{escola}</option>
                            ))}
                        </select>
                    </div>

                    <div className="flex flex-col gap-2">
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Turma</label>
                        <select
                            value={filtros.turma}
                            onChange={(e) => handleChange('turma', e.target.value)}
                            className="px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-700 bg-white focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100 transition-all"
                        >
                            <option value="">Todas as turmas</option>
                            {turmas.map(turma => (
                                <option key={turma} value={turma}>{turma}</option>
                            ))}
                        </select>
                    </div>

                    <div className="flex flex-col gap-2">
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Data das provas</label>
                        <input
                            type="date"
                            value={filtros.dataInicio}
                            onChange={(e) => handleChange('dataInicio', e.target.value)}
                            className="px-3 py-2 border border-gray-200 rounded-lg text-sm text-gray-700 bg-white focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100 transition-all"
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
