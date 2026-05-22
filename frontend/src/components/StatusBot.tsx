import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Bot, CheckCircle2, PauseCircle } from 'lucide-react';
import { buscarStatus, conectarStreamBot, type BotStreamState, type Status } from '../service/api';

const CHAVE_DATA = 'statusbot:ultima_data';
const CHAVE_CORRIGIDOS = 'statusbot:corrigidos_sessao';

function obterHoje(): string {
    return new Date().toISOString().split('T')[0];
}

function carregarCorrigidosSessao(): number {
    const dataSalva = localStorage.getItem(CHAVE_DATA);
    const hoje = obterHoje();

    if (dataSalva !== hoje) {
        // Dia mudou — zera tudo
        localStorage.setItem(CHAVE_DATA, hoje);
        localStorage.setItem(CHAVE_CORRIGIDOS, '0');
        return 0;
    }

    return Number(localStorage.getItem(CHAVE_CORRIGIDOS) ?? 0);
}

function salvarCorrigidosSessao(valor: number): void {
    localStorage.setItem(CHAVE_DATA, obterHoje());
    localStorage.setItem(CHAVE_CORRIGIDOS, String(valor));
}

const FORMATADOR_DATA_BR = new Intl.DateTimeFormat('pt-BR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    timeZone: 'America/Sao_Paulo'
});

function interpretarData(valor: string): Date | null {
    const texto = valor.trim();
    if (!texto) return null;

    // Alguns eventos chegam sem informacao de fuso. Nestes casos tratamos o horario como UTC.
    const semFuso = texto.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2})(?:\.\d{1,6})?)?$/);
    if (semFuso) {
        const [, ano, mes, dia, hora, minuto, segundo = '00'] = semFuso;
        const dataUtc = new Date(
            Date.UTC(Number(ano), Number(mes) - 1, Number(dia), Number(hora), Number(minuto), Number(segundo))
        );

        if (!Number.isNaN(dataUtc.getTime())) {
            return dataUtc;
        }
    }

    const data = new Date(texto);
    if (Number.isNaN(data.getTime())) {
        return null;
    }

    return data;
}

function formatarData(valor: string | null): string {
    if (!valor) return 'Sem informacao';

    const data = interpretarData(valor);
    if (!data) return valor;

    return FORMATADOR_DATA_BR.format(data);
}


function criarStatusInicial(): Status {
    return {
        status: 'idle',
        timestamp: new Date().toISOString(),
        bot_ativo: false,
        arquivo_atual: null,
        progresso: 0,
        corrigidos_sessao: carregarCorrigidosSessao(), // 👈 lê do localStorage
        ultima_correcao_sessao: null,
        ultima_atualizacao: null,
        total_registros_9ano: 0,
        total_registros_5ano: 0,
        database: 'disconnected'
    };
}

function aplicarStreamNoStatus(anterior: Status | null, stream: BotStreamState): Status {
    const base = anterior ?? criarStatusInicial();

    // Usa o valor da API, mas persiste localmente para sobreviver ao F5
    const corrigidosDaApi = Number(stream.total_corrected || 0);
    salvarCorrigidosSessao(corrigidosDaApi);

    return {
        ...base,
        status: stream.status,
        bot_ativo: stream.status === 'running',
        arquivo_atual: stream.current_file,
        progresso: Number(stream.progress || 0),
        corrigidos_sessao: corrigidosDaApi,
        ultima_correcao_sessao: stream.last_correction,
        ultima_atualizacao: stream.last_correction ?? base.ultima_atualizacao,
        timestamp: new Date().toISOString(),
    };
}

function aplicarApiNoStatus(anterior: Status | null, respostaApi: Status): Status {
    const corrigidosPersistidos = carregarCorrigidosSessao();
    const corrigidosApi = Number(respostaApi.corrigidos_sessao || 0);
    const base = {
        ...respostaApi,
        corrigidos_sessao: Math.max(corrigidosApi, corrigidosPersistidos),
    };

    if (anterior?.status !== 'running') {
        return base;
    }

    return {
        ...base,
        status: anterior.status,
        bot_ativo: true,
        arquivo_atual: anterior.arquivo_atual,
        progresso: anterior.progresso,
        corrigidos_sessao: Math.max(base.corrigidos_sessao, Number(anterior.corrigidos_sessao || 0)),
        ultima_correcao_sessao: anterior.ultima_correcao_sessao,
        timestamp: anterior.timestamp,
    };
}

export function StatusBot() {
    const [status, setStatus] = useState<Status | null>(null);
    const [loading, setLoading] = useState(true);
    const [aoVivo, setAoVivo] = useState(false);
    const [erro, setErro] = useState<string | null>(null);
    const ultimoStatusRef = useRef<Status['status'] | null>(null);
    const statusRef = useRef<Status | null>(null);
    

    const notificarFimDaLeitura = useCallback((proximoStatus: Status['status']) => {
        const statusAnterior = ultimoStatusRef.current;

        if (statusAnterior === 'running' && proximoStatus === 'idle') {
            window.dispatchEvent(new CustomEvent('bot:leitura-finalizada'));
        }

        ultimoStatusRef.current = proximoStatus;
    }, []);

    const carregarStatus = useCallback(async (emSegundoPlano = false) => {
        if (!emSegundoPlano) {
            setLoading(true);
        }

        try {
            const response = await buscarStatus();
            const proximoStatus = aplicarApiNoStatus(statusRef.current, response);
            statusRef.current = proximoStatus;
            setStatus(proximoStatus);
            notificarFimDaLeitura(proximoStatus.status);
            setErro(null);
        } catch {
            setErro('Nao foi possivel atualizar o status agora.');
        } finally {
            setLoading(false);
        }
    }, [notificarFimDaLeitura]);

    useEffect(() => {
        void carregarStatus();

        const desconectar = conectarStreamBot(
            (dados) => {
                const proximoStatus = aplicarStreamNoStatus(statusRef.current, dados);
                statusRef.current = proximoStatus;
                setStatus(proximoStatus);
                notificarFimDaLeitura(proximoStatus.status);
                setAoVivo(true);
                setErro(null);
            },
            () => {
                setAoVivo(false);
            }
        );

        const intervalo = setInterval(() => {
            void carregarStatus(true);
        }, 30000);

        return () => {
            clearInterval(intervalo);
            desconectar();
        };
    }, [carregarStatus, notificarFimDaLeitura]);

    const progressoAtual = useMemo(() => {
        if (!status) return 0;
        return Math.max(0, Math.min(100, Number(status.progresso || 0)));
    }, [status]);

    if (loading) {
        return (
            <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-200">
                <div className="animate-pulse space-y-3">
                    <div className="h-5 bg-gray-200 rounded w-56"></div>
                    <div className="h-4 bg-gray-200 rounded w-40"></div>
                    <div className="h-2 bg-gray-200 rounded w-full"></div>
                </div>
            </div>
        );
    }

    if (!status) {
        return (
            <div className="bg-white rounded-xl p-5 shadow-sm border border-red-200">
                <p className="text-red-700 font-medium">Nao foi possivel carregar o status do bot.</p>
            </div>
        );
    }



    return (
        <section className="bg-white rounded-xl p-5 shadow-sm border border-gray-200">
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-gray-900">
                    <Bot size={20} />
                    <h2 className="text-lg font-semibold">Status Atual do Bot</h2>
                </div>
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-2">

                <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium border ${
                    aoVivo
                        ? 'bg-cyan-100 text-cyan-700 border-cyan-200'
                        : 'bg-gray-100 text-gray-700 border-gray-200'
                }`}>
                    <Activity size={14} className={aoVivo ? 'animate-pulse' : ''} />
                    {aoVivo ? 'Ao vivo' : 'Sem stream'}
                </span>

                <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium border ${
                    status.bot_ativo
                        ? 'bg-emerald-100 text-emerald-700 border-emerald-200'
                        : 'bg-gray-100 text-gray-700 border-gray-200'
                }`}>
                    {status.bot_ativo ? <CheckCircle2 size={14} /> : <PauseCircle size={14} />}
                    {status.bot_ativo ? 'Bot ativo' : 'Bot inativo'}
                </span>
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                <div className="rounded-lg border border-gray-200 p-3 bg-gray-50">
                    <p className="text-xs text-gray-500 uppercase tracking-wide">Arquivo atual</p>
                    <p className="text-sm font-medium text-gray-800 mt-1 truncate" title={status.arquivo_atual ?? 'Nenhum'}>
                        {status.arquivo_atual ?? 'Nenhum'}
                    </p>
                </div>

                <div className="rounded-lg border border-gray-200 p-3 bg-gray-50">
                    <p className="text-xs text-gray-500 uppercase tracking-wide">Ultima atualizacao</p>
                    <p className="text-sm font-medium text-gray-800 mt-1">{(status.ultima_atualizacao)}</p>
                </div>

                <div className="rounded-lg border border-gray-200 p-3 bg-gray-50">
                    <p className="text-xs text-gray-500 uppercase tracking-wide">Corrigidos na sessao</p>
                    <p className="text-sm font-medium text-gray-800 mt-1">{(status.corrigidos_sessao)}</p>
                </div>

            </div>

            <div className="mt-4">
                <div className="flex items-center justify-between text-sm mb-1">
                    <span className="text-gray-600">Progresso</span>
                    <span className="font-medium text-gray-800">{progressoAtual}%</span>
                </div>
                <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-500"
                        style={{ width: `${progressoAtual}%` }}
                    />
                </div>
            </div>

            <div className="mt-4 text-sm text-gray-600">
                <p>Total de cartões processados: {(status.total_registros_5ano ?? 0) + (status.total_registros_9ano ?? 0)}</p>
                <p className="text-xs text-gray-500 mt-1">Hora ao vivo: {formatarData(status.timestamp)}</p>
            </div>

            {erro && <p className="mt-3 text-sm text-amber-700">{erro}</p>}
        </section>
    );
}
