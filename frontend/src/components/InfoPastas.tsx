import { useEffect, useState } from 'react';
import { api, type PastaResponse } from '../service/api';

interface Props {
    ano: '5' | '9' | 'geral';
    refreshToken?: number;
}

export function InfoPastas({ ano, refreshToken = 0 }: Props) {
    const [pasta5, setPasta5] = useState<PastaResponse | null>(null);
    const [pasta9, setPasta9] = useState<PastaResponse | null>(null);
    const [loading, setLoading] = useState(true);

    const renderPasta = (pasta: PastaResponse | null) => {
        if (!pasta) return null;

        return (
            <div className="bg-white rounded-xl p-2 shadow-sm py-1">
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                            📁 {pasta.descricao}
                        </h3>
                        <p className="text-sm text-gray-600 mt-1">
                            Total de arquivos: <span className="font-semibold text-blue-600">{pasta.total_registros}</span>
                        </p>
                    </div>
                </div>
            </div>
        );
    };

    

    useEffect(() => {
        async function carregarPastas() {
            setLoading(true);
            try {
                if (ano === 'geral') {
                    const [resp5, resp9] = await Promise.all([
                        api.getPasta5Ano(),
                        api.getPasta9Ano()
                    ]);
                    setPasta5(resp5);
                    setPasta9(resp9);
                } else if (ano === '9') {
                    const resp = await api.getPasta9Ano();
                    setPasta9(resp);
                    setPasta5(null);
                } else {
                    const resp = await api.getPasta5Ano();
                    setPasta5(resp);
                    setPasta9(null);
                }
            } catch (error) {
                console.error('Erro ao carregar pastas:', error);
            } finally {
                setLoading(false);
            }
        }
        carregarPastas();
    }, [ano, refreshToken]);

if (loading) {
        return (
            <div className="bg-white rounded-xl p-6 shadow-sm mb-6">
                <div className="animate-pulse flex space-x-4">
                    <div className="flex-1 space-y-3">
                        <div className="h-2 bg-gray-200 rounded w-3/4"></div>
                        <div className="h-2 bg-gray-200 rounded w-1/2"></div>
                    </div>
                </div>
            </div>
        );
    }
        

    if (ano === 'geral' ? (pasta5 && pasta9) : (ano === '9' ? pasta9 : pasta5))
        return (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {ano === 'geral' ? (
                    <>
                        {renderPasta(pasta5)}
                        {renderPasta(pasta9)}
                    </>
                ) : (
                    renderPasta(ano === '9' ? pasta9 : pasta5)
                )}
            </div>
        );
}
