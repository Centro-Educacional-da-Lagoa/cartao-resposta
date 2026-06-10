import { useCallback, useEffect, useMemo, useState } from 'react';
import { ANOS, api, type Aluno, type Ano, type AnoFiltro } from '../service/api';
import { CardsEstatisticas } from './CardsEstatisticas';
import { Graficos } from './Graficos';
import { InfoPastas } from './InfoPastas';
import { StatusBot } from './StatusBot';
import { TabelaAlunos } from './TabelaAlunos';

interface Props {
    ano: AnoFiltro;
}

const ALUNOS_INICIAIS: Record<Ano, Aluno[]> = {
    '4': [],
    '5': [],
    '8': [],
    '9': [],
};

export function Dashboard({ ano }: Props) {
    const [alunosPorAno, setAlunosPorAno] = useState<Record<Ano, Aluno[]>>(ALUNOS_INICIAIS);
    const [loading, setLoading] = useState(true);
    const [refreshToken, setRefreshToken] = useState(0);

    const carregarDados = useCallback(async () => {
        setLoading(true);
        try {
            const anosParaCarregar = ano === 'geral' ? ANOS : [ano];
            const respostas = await Promise.all(
                anosParaCarregar.map(async (anoAtual) => ({
                    ano: anoAtual,
                    resposta: await api.getAlunos(anoAtual),
                }))
            );

            setAlunosPorAno((estadoAtual) => {
                const proximoEstado = ano === 'geral' ? { ...ALUNOS_INICIAIS } : { ...estadoAtual };
                for (const item of respostas) {
                    proximoEstado[item.ano] = item.resposta.alunos || [];
                }
                return proximoEstado;
            });
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
        return () => window.removeEventListener('bot:leitura-finalizada', handleLeituraFinalizada);
    }, [carregarDados]);

    const todosAlunos = useMemo(
        () => ano === 'geral'
            ? ANOS.flatMap((anoAtual) => alunosPorAno[anoAtual])
            : alunosPorAno[ano],
        [ano, alunosPorAno]
    );

    if (loading) {
        return (
            <div className="flex min-h-screen items-center justify-center">
                <div className="text-center">
                    <div className="mb-4 inline-block h-12 w-12 animate-spin rounded-full border-b-2 border-blue-600" />
                    <p className="text-lg text-gray-600">Carregando dados...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <StatusBot />
            <InfoPastas ano={ano} refreshToken={refreshToken} />

            {ano === 'geral' ? (
                <div className="space-y-8">
                    {ANOS.map((anoAtual) => (
                        <section key={anoAtual} className="space-y-4">
                            <h3 className="text-xl font-semibold text-gray-700">
                                Estatísticas - {anoAtual}º Ano
                            </h3>
                            <CardsEstatisticas ano={anoAtual} alunos={alunosPorAno[anoAtual]} />
                            <Graficos ano={anoAtual} alunos={alunosPorAno[anoAtual]} />
                        </section>
                    ))}
                </div>
            ) : (
                <>
                    <CardsEstatisticas ano={ano} alunos={todosAlunos} />
                    <Graficos ano={ano} alunos={todosAlunos} />
                </>
            )}

            <TabelaAlunos alunos={todosAlunos} ano={ano} />
        </div>
    );
}
