import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, type Aluno } from '../service/api';
import { CardsEstatisticas } from './CardsEstatisticas';
import { Graficos } from './Graficos';
import { TabelaAlunos } from './TabelaAlunos';
import { InfoPastas } from './InfoPastas';
import { StatusBot } from './StatusBot';

interface Props {
    ano: '5' | '9' | 'geral';
}

export function Dashboard({ ano }: Props) {
    const [alunos5, setAlunos5] = useState<Aluno[]>([]);
    const [alunos9, setAlunos9] = useState<Aluno[]>([]);
    const [loading, setLoading] = useState(true);
    const [refreshToken, setRefreshToken] = useState(0);

    const carregarDados = useCallback(async () => {
        setLoading(true);
        try {
            if (ano === 'geral') {
                const [resp5, resp9] = await Promise.all([
                    api.getAlunos5ano(),
                    api.getAlunos9Ano()
                ]);
                setAlunos5(resp5.alunos || []);
                setAlunos9(resp9.alunos || []);
            } else if (ano === '9') {
                const resp = await api.getAlunos9Ano();
                setAlunos9(resp.alunos || []);
                setAlunos5([]);
            } else {
                const resp = await api.getAlunos5ano();
                setAlunos5(resp.alunos || []);
                setAlunos9([]);
            }
        } catch (error) {
            console.error('Erro ao carregar dados:', error);
        } finally {
            setLoading(false);
        }
    }, [ano]);

    useEffect(() => {
        void carregarDados();
    }, [carregarDados]);

    useEffect(() => {
        const handleLeituraFinalizada = () => {
            void carregarDados();
            setRefreshToken((valorAtual) => valorAtual + 1);
        };

        window.addEventListener('bot:leitura-finalizada', handleLeituraFinalizada);

        return () => {
            window.removeEventListener('bot:leitura-finalizada', handleLeituraFinalizada);
        };
    }, [carregarDados]);

    const todosAlunos = useMemo(() => {
        if (ano === 'geral') {
            return [...alunos5, ...alunos9];
        } else if (ano === '9') {
            return alunos9;
        } else {
            return alunos5;
        }
    }, [ano, alunos5, alunos9]);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                    <p className="text-gray-600 text-lg">Carregando dados...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Status do Bot */}
            <StatusBot />

            {/* Informações das Pastas */}
            <InfoPastas ano={ano} refreshToken={refreshToken} />

            {/* Cards de Estatísticas */}
            {ano !== 'geral' && <CardsEstatisticas ano={ano} alunos={todosAlunos} />}

            {ano === 'geral' && (
                <div className="space-y-8">
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-gray-700">📊 Estatísticas - 9º Ano</h3>
                        <CardsEstatisticas ano="9" alunos={alunos9} />
                    </div>
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-gray-700">📊 Estatísticas - 5º Ano</h3>
                        <CardsEstatisticas ano="5" alunos={alunos5} />
                    </div>
                </div>
            )}

            {/* Gráficos */}
            {ano !== 'geral' && <Graficos ano={ano} alunos={todosAlunos} />}

            {ano === 'geral' && (
                <div className="space-y-8">
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-gray-700">📈 Gráficos - 9º Ano</h3>
                        <Graficos ano="9" alunos={alunos9} />
                    </div>
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-gray-700">📈 Gráficos - 5º Ano</h3>
                        <Graficos ano="5" alunos={alunos5} />
                    </div>
                </div>
            )}

            {/* Tabela de Alunos */}
            <TabelaAlunos alunos={todosAlunos} ano={ano} />
        </div>
    );
}

