import { useEffect, useMemo, useState } from 'react';
import { BarChart3, LayoutDashboard, Loader2, LogOut, Menu, School2, X } from 'lucide-react';
import { Dashboard } from './components/Dashboard';
import { AuthPage } from './components/auth/AuthPage';
import { api, buscarStatus, type Status, type Usuario } from './service/api';

type Ano = '5' | '9' | 'geral';

type NavItem = {
  valor: Ano;
  titulo: string;
  descricao: string;
  Icone: React.ComponentType<{ size?: number; className?: string }>;
};

const NAV_ITEMS: NavItem[] = [
  {
    valor: 'geral',
    titulo: 'Visao Geral',
    descricao: 'Visão geral de estatísticas e dados combinados de todas as turmas',
    Icone: LayoutDashboard
  },
  {
    valor: '9',
    titulo: '9° Ano',
    descricao: 'Filtra dados e estatisticas para a turma do 9° ano',
    Icone: School2
  },
  {
    valor: '5',
    titulo: '5° Ano',
    descricao: 'Filtra dados e estatisticas para a turma do 5° ano',
    Icone: School2
  }
];

function App() {
  const [usuario, setUsuario] = useState<Usuario | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [anoSelecionado, setAnoSelecionado] = useState<Ano>('geral');
  const [menuAberto, setMenuAberto] = useState(false);
  const [statusSistema, setStatusSistema] = useState<Status['status']>('idle');

  useEffect(() => {
    let ativo = true;

    const carregarUsuario = async () => {
      try {
        const usuarioAtual = await api.getUsuarioAtual();
        if (ativo) {
          setUsuario(usuarioAtual);
        }
      } catch {
        if (ativo) {
          setUsuario(null);
        }
      } finally {
        if (ativo) {
          setAuthLoading(false);
        }
      }
    };

    void carregarUsuario();

    return () => {
      ativo = false;
    };
  }, []);

  useEffect(() => {
    if (!menuAberto) {
      document.body.style.overflow = '';
      return;
    }

    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setMenuAberto(false);
      }
    };

    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', handleEsc);

    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', handleEsc);
    };
  }, [menuAberto]);

  useEffect(() => {
    if (!usuario) {
      return;
    }

    let ativo = true;

    const carregarStatusSistema = async () => {
      try {
        const status = await buscarStatus();
        if (!ativo) return;
        setStatusSistema(status.status);
      } catch {
        if (!ativo) return;
        setStatusSistema('error');
      }
    };

    void carregarStatusSistema();

    const intervalo = setInterval(() => {
      void carregarStatusSistema();
    }, 30000);

    return () => {
      ativo = false;
      clearInterval(intervalo);
    };
  }, [usuario]);

  const itemAtivo = useMemo(
    () => NAV_ITEMS.find((item) => item.valor === anoSelecionado) ?? NAV_ITEMS[0],
    [anoSelecionado]
  );

  const handleSelecionarAno = (ano: Ano) => {
    setAnoSelecionado(ano);
    setMenuAberto(false);
  };

  const handleLogout = async () => {
    try {
      await api.logout();
    } finally {
      setUsuario(null);
      setMenuAberto(false);
    }
  };

  if (authLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-white text-neutral-700">
        <div className="flex items-center gap-3 border border-neutral-200 bg-white px-4 py-3">
          <Loader2 size={18} className="animate-spin text-neutral-950" />
          <span className="text-sm font-semibold">Carregando sessao...</span>
        </div>
      </div>
    );
  }

  if (!usuario) {
    return <AuthPage onAuthenticated={setUsuario} />;
  }

  return (
    <div className="app-shell min-h-screen overflow-x-hidden text-slate-900 bg-gray-300">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-28 -right-20 h-96 w-96 rounded-full bg-cyan-400/20 blur-3xl" />
        <div className="absolute bottom-0 -left-28 h-80 w-80 rounded-full bg-teal-500/20 blur-3xl" />
      </div>

      <div className="relative flex min-h-screen">
        {menuAberto && (
          <button
            type="button"
            aria-label="Fechar menu"
            onClick={() => setMenuAberto(false)}
            className="fixed inset-0 z-30 bg-slate-950/45 backdrop-blur-[1px]"
          />
        )}

        <aside
          className={`fixed inset-y-0 left-0 z-40 w-[86vw] max-w-80 transform border-r border-white/15 bg-gray-500/60 text-slate-100 shadow-2xl backdrop-blur-xl transition-transform duration-300 sm:max-w-sm ${
            menuAberto ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          <div className="flex h-full flex-col px-5 py-6">
            <div className="flex items-start justify-between gap-3 border-b border-white/10 pb-5">
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 to-teal-500 text-slate-950 shadow-lg shadow-cyan-500/30">
                  <BarChart3 size={24} />
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Centro Educacional</p>
                  <h1 className="text-lg font-semibold leading-tight text-white">Painel de Correcao</h1>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setMenuAberto(false)}
                className="rounded-lg border border-white/20 p-2 text-slate-200 hover:bg-white/10"
              >
                <X size={16} />
              </button>
            </div>

            <nav className="mt-6 space-y-2">
              {NAV_ITEMS.map((item) => {
                const ativo = anoSelecionado === item.valor;
                const Icone = item.Icone;

                return (
                  <button
                    key={item.valor}
                    type="button"
                    onClick={() => handleSelecionarAno(item.valor)}
                    className={`group flex w-full items-center gap-3 rounded-2xl border px-4 py-3 text-left transition-all duration-200 ${
                      ativo
                        ? 'border-cyan-300/60 bg-cyan-400/10 shadow-lg shadow-cyan-500/10'
                        : 'border-white/10 hover:border-white/35 hover:bg-white/5'
                    }`}
                  >
                    <span
                      className={`flex h-9 w-9 items-center justify-center rounded-xl transition-colors ${
                        ativo ? 'bg-cyan-300/20 text-cyan-200' : 'bg-white/10 text-slate-300 group-hover:text-white'
                      }`}
                    >
                      <Icone size={18} />
                    </span>

                    <span>
                      <span className="block text-sm font-semibold text-white">{item.titulo}</span>
                      <span className="block text-xs text-slate-400">{item.descricao}</span>
                    </span>
                  </button>
                );
              })}
            </nav>
          </div>
        </aside>

        <div className="flex min-h-screen flex-1 flex-col pt-[82px] sm:pt-[90px]">
          <header className="fixed inset-x-0 top-0 z-20 border-b border-white/60 bg-white/85 backdrop-blur-xl">
            <div className="mx-auto flex w-full max-w-[1440px] items-center justify-between px-4 py-4 sm:px-6 lg:px-10">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  aria-label="Abrir menu"
                  aria-expanded={menuAberto}
                  onClick={() => setMenuAberto(true)}
                  className="rounded-xl border border-slate-200 bg-white p-2 text-slate-700 shadow-sm"
                >
                  <Menu size={18} />
                </button>

                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">Painel ativo</p>
                  <h2 className="text-lg font-semibold text-slate-900 sm:text-xl">{itemAtivo.titulo}</h2>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="hidden items-center gap-3 rounded-2xl border border-slate-200/80 bg-white/80 px-4 py-2 shadow-sm sm:flex">
                  <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Status do sistema</p>
                  <div className="flex items-center justify-end gap-2" aria-label="Status do bot">
                    <span
                      className={`h-2.5 w-2.5 rounded-full ${
                        statusSistema === 'running'
                          ? 'bg-emerald-500 shadow-[0_0_0_3px_rgba(16,185,129,0.18)]'
                          : statusSistema === 'idle'
                            ? 'bg-amber-400 shadow-[0_0_0_3px_rgba(251,191,36,0.24)]'
                            : 'bg-red-500 shadow-[0_0_0_3px_rgba(239,68,68,0.2)]'
                      }`}
                      title={
                        statusSistema === 'running'
                          ? 'Rodando'
                          : statusSistema === 'idle'
                            ? 'Parado'
                            : 'Erro'
                      }
                    />
                  </div>
                </div>

                <div className="hidden max-w-56 text-right md:block">
                  <p className="truncate text-sm font-semibold text-slate-900">{usuario.nome}</p>
                  <p className="truncate text-xs text-slate-500">{usuario.email}</p>
                </div>

                <button
                  type="button"
                  onClick={handleLogout}
                  className="rounded-xl border border-slate-200 bg-white p-2 text-slate-700 shadow-sm transition hover:bg-slate-50"
                  aria-label="Sair"
                  title="Sair"
                >
                  <LogOut size={18} />
                </button>
              </div>
            </div>
          </header>

          <main className="mx-auto w-full max-w-[1440px] flex-1 px-4 py-6 sm:px-6 lg:px-10 lg:py-8">
            <div className="rounded-[30px] border border-white/70 bg-white/80 p-4 shadow-[0_24px_65px_-40px_rgba(15,23,42,0.45)] backdrop-blur sm:p-6">
              <Dashboard ano={anoSelecionado} />
            </div>
          </main>

          <footer className="border-t border-white/70 bg-white/80">
            <div className="mx-auto flex w-full max-w-[1440px] items-center justify-between gap-3 px-4 py-4 text-xs text-slate-500 sm:px-6 lg:px-10">
              <p>2026 Centro Educacional da Lagoa</p>
              <p>Dashboard de Correcao de Cartoes</p>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}

export default App;
