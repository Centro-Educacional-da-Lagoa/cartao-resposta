import { useEffect, useState } from 'react';
import { ANOS, api, type Ano, type AnoFiltro, type PastaResponse } from '../service/api';

interface Props {
    ano: AnoFiltro;
    refreshToken?: number;
}

const PASTAS_INICIAIS: Record<Ano, PastaResponse | null> = {
    '4': null,
    '5': null,
    '8': null,
    '9': null,
};

export function InfoPastas({ ano, refreshToken = 0 }: Props) {
    const [pastas, setPastas] = useState<Record<Ano, PastaResponse | null>>(PASTAS_INICIAIS);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function carregarPastas() {
            setLoading(true);
            try {
                const anosParaCarregar = ano === 'geral' ? ANOS : [ano];
                const respostas = await Promise.all(
                    anosParaCarregar.map(async (anoAtual) => ({
                        ano: anoAtual,
                        resposta: await api.getPasta(anoAtual),
                    }))
                );
                setPastas((estadoAtual) => {
                    const proximoEstado = ano === 'geral' ? { ...PASTAS_INICIAIS } : { ...estadoAtual };
                    for (const item of respostas) {
                        proximoEstado[item.ano] = item.resposta;
                    }
                    return proximoEstado;
                });
            } catch (error) {
                console.error('Erro ao carregar pastas:', error);
            } finally {
                setLoading(false);
            }
        }

        void carregarPastas();
    }, [ano, refreshToken]);

    if (loading) {
        return (
            <div className="mb-6 rounded-xl bg-white p-6 shadow-sm">
                <div className="h-2 w-3/4 animate-pulse rounded bg-gray-200" />
            </div>
        );
    }

    const anosVisiveis = ano === 'geral' ? ANOS : [ano];

    return (
        <div className="mb-6 grid grid-cols-1 gap-6 md:grid-cols-2">
            {anosVisiveis.map((anoAtual) => {
                const pasta = pastas[anoAtual];
                if (!pasta) return null;

                return (
                    <div key={anoAtual} className="rounded-xl bg-white p-3 shadow-sm">
                        <h3 className="text-lg font-semibold text-gray-800">{pasta.descricao}</h3>
                        <p className="mt-1 text-sm text-gray-600">
                            Total de arquivos:{' '}
                            <span className="font-semibold text-blue-600">{pasta.total_registros}</span>
                        </p>
                    </div>
                );
            })}
        </div>
    );
}
