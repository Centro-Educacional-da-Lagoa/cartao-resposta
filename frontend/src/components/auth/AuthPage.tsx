import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from 'react';
import {
  ArrowRight,
  Loader2,
  Lock,
  LogIn,
  Mail,
  User,
  UserPlus,
  type LucideIcon,
} from 'lucide-react';
import { api, type Usuario } from '../../service/api';

type AuthMode = 'login' | 'register';

type GoogleCredentialResponse = {
  credential?: string;
};

type GoogleButtonOptions = {
  theme: 'outline';
  size: 'large';
  type: 'standard';
  text: 'signin_with' | 'signup_with' | 'continue_with';
  shape: 'rectangular';
  width: number;
  locale?: string;
};

type GoogleAccounts = {
  accounts: {
    id: {
      initialize: (options: {
        client_id: string;
        callback: (response: GoogleCredentialResponse) => void;
      }) => void;
      renderButton: (element: HTMLElement, options: GoogleButtonOptions) => void;
    };
  };
};

declare global {
  interface Window {
    google?: GoogleAccounts;
  }
}

interface AuthPageProps {
  onAuthenticated: (usuario: Usuario) => void;
}

type AuthInputProps = {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  Icon: LucideIcon;
  autoComplete: string;
  type?: string;
  minLength?: number;
  maxLength?: number;
  required?: boolean;
};

const GOOGLE_SCRIPT_URL = 'https://accounts.google.com/gsi/client';

function getApiErrorMessage(error: unknown): string {
  const apiError = error as {
    response?: {
      data?: {
        message?: string | string[];
      };
    };
  };
  const message = apiError.response?.data?.message;

  if (Array.isArray(message)) {
    return message[0] ?? 'Não foi possível concluir a autenticação.';
  }

  return message ?? 'Não foi possível concluir a autenticação.';
}

function AuthInput({
  id,
  label,
  value,
  onChange,
  Icon,
  autoComplete,
  type = 'text',
  minLength,
  maxLength,
  required = false,
}: AuthInputProps) {
  return (
    <label
      htmlFor={id}
      className="group relative block border-b border-neutral-300 transition-colors duration-300 focus-within:border-neutral-950"
    >
      <Icon
        size={18}
        className="pointer-events-none absolute left-0 top-1/2 z-10 -translate-y-1/2 text-neutral-400 transition-colors duration-300 group-focus-within:text-neutral-950"
      />
      <input
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="peer h-16 w-full bg-transparent pl-9 pr-2 pt-5 text-[15px] font-medium text-neutral-950 outline-none transition-colors duration-300 placeholder:text-transparent selection:bg-neutral-950 selection:text-white"
        type={type}
        placeholder=" "
        autoComplete={autoComplete}
        minLength={minLength}
        maxLength={maxLength}
        required={required}
      />
      <span className="pointer-events-none absolute left-9 top-2 translate-y-0 text-[11px] font-semibold uppercase tracking-[0.18em] text-neutral-950 transition-all duration-300 ease-out peer-placeholder-shown:top-1/2 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:text-sm peer-placeholder-shown:font-medium peer-placeholder-shown:normal-case peer-placeholder-shown:tracking-normal peer-placeholder-shown:text-neutral-500 peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-[11px] peer-focus:font-semibold peer-focus:uppercase peer-focus:tracking-[0.18em] peer-focus:text-neutral-950">
        {label}
      </span>
    </label>
  );
}

export function AuthPage({ onAuthenticated }: AuthPageProps) {
  const [mode, setMode] = useState<AuthMode>('login');
  const [nome, setNome] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [erro, setErro] = useState('');
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const googleButtonRef = useRef<HTMLDivElement | null>(null);

  const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined;
  const isRegister = mode === 'register';

  const copy = useMemo(
    () =>
      isRegister
        ? {
            description: 'Registre-se para acompanhar a correção dos cartões-respostas.',
            action: 'Se registrar',
          }
        : {
            description: 'Entre para acessar os resultados escolares e acompanhar a correção dos cartões-respostas.',
            action: 'Acessar',
          },
    [isRegister],
  );

  const handleGoogleCredential = useCallback(
    async (credential: string) => {
      setErro('');
      setGoogleLoading(true);
      try {
        const response = await api.loginGoogle(credential);
        onAuthenticated(response.user);
      } catch (error) {
        setErro(getApiErrorMessage(error));
      } finally {
        setGoogleLoading(false);
      }
    },
    [onAuthenticated],
  );

  useEffect(() => {
    if (!googleClientId || !googleButtonRef.current) {
      return;
    }

    let active = true;
    const buttonElement = googleButtonRef.current;

    const renderGoogleButton = () => {
      if (!active || !window.google || !buttonElement) {
        return;
      }

      buttonElement.innerHTML = '';
      window.google.accounts.id.initialize({
        client_id: googleClientId,
        callback: (response) => {
          if (response.credential) {
            void handleGoogleCredential(response.credential);
            return;
          }

          setErro('Não foi possível validar a conta Google.');
        },
      });
      window.google.accounts.id.renderButton(buttonElement, {
        theme: 'outline',
        size: 'large',
        type: 'standard',
        text: 'continue_with',
        shape: 'rectangular',
        width: Math.min(buttonElement.offsetWidth || 360, 420),
        locale: 'pt-BR',
      });
    };

    if (window.google) {
      renderGoogleButton();
      return () => {
        active = false;
      };
    }

    const existingScript = document.querySelector<HTMLScriptElement>(
      `script[src="${GOOGLE_SCRIPT_URL}"]`,
    );

    if (existingScript) {
      existingScript.addEventListener('load', renderGoogleButton);
      return () => {
        active = false;
        existingScript.removeEventListener('load', renderGoogleButton);
      };
    }

    const script = document.createElement('script');
    script.src = GOOGLE_SCRIPT_URL;
    script.async = true;
    script.defer = true;
    script.addEventListener('load', renderGoogleButton);
    script.addEventListener('error', () => setErro('Não foi possível carregar o Google Sign-In.'));
    document.head.appendChild(script);

    return () => {
      active = false;
      script.removeEventListener('load', renderGoogleButton);
    };
  }, [googleClientId, handleGoogleCredential, isRegister]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setErro('');
    setLoading(true);

    try {
      const response = isRegister
        ? await api.registrar({ nome, email, password })
        : await api.login({ email, password });

      onAuthenticated(response.user);
    } catch (error) {
      setErro(getApiErrorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  const selecionarModo = (nextMode: AuthMode) => {
    setErro('');
    setMode(nextMode);
  };

  return (
    <main className="auth-page auth-surface relative min-h-screen overflow-hidden text-neutral-950">
      <div className="pointer-events-none absolute inset-0" aria-hidden="true">
        <div className="absolute -left-24 top-12 h-64 w-64 rounded-full border border-neutral-300/70" />
        <div className="absolute right-[-7rem] top-[-7rem] h-80 w-80 rounded-full border border-neutral-200" />
        <div className="absolute bottom-16 left-[9%] h-28 w-28 rotate-45 border border-neutral-300/80" />
        <div className="absolute bottom-0 right-[12%] h-px w-56 bg-neutral-300" />
      </div>

      <section className="relative z-10 grid min-h-screen w-full grid-cols-1 bg-white/85 backdrop-blur md:grid-cols-[1.08fr_0.92fr]">
        <aside className="relative hidden overflow-hidden border-r border-neutral-200 p-10 md:flex md:flex-col md:justify-between lg:p-16 xl:p-20">
          <div className="pointer-events-none absolute -right-32 top-20 h-72 w-72 rounded-full border border-neutral-300/80" />
          <div className="pointer-events-none absolute bottom-28 right-14 h-20 w-20 rotate-45 border border-neutral-200" />

          <div className="relative">
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-neutral-500">
              Centro Educacional
            </p>
            <h1 className="mt-5 max-w-[11ch] text-6xl font-black leading-[0.92] text-neutral-950 lg:text-7xl">
              Painel de Correção
            </h1>
          </div>

          <div className="relative max-w-sm border-t border-neutral-200 pt-6 text-sm leading-6 text-neutral-600">
            <p>Acesso protegido para consulta dos resultados escolares.</p>
          </div>
        </aside>

        <div className="flex min-h-screen items-center px-5 py-10 sm:px-10 lg:px-16 xl:px-20">
          <div className="mx-auto w-full max-w-lg">
            <div className="mb-10 md:hidden">
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-neutral-500">
                Centro Educacional
              </p>
              <h1 className="mt-3 text-5xl font-black leading-none text-neutral-950">
                Painel de Correção
              </h1>
            </div>

            <div className="mb-9 grid gap-7">
              <div className="flex items-start justify-between gap-5">
                  <p className="max-w-sm text-sm leading-6 text-neutral-600">{copy.description}</p>
                </div>


              <div
                className="relative grid grid-cols-2 border border-neutral-300 p-1"
                role="group"
                aria-label="Alternar modo de autenticação"
              >
                <span
                  className="absolute bottom-1 left-1 top-1 w-[calc(50%-4px)] bg-neutral-950 transition-transform duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]"
                  style={{ transform: isRegister ? 'translateX(calc(100% + 4px))' : 'translateX(0)' }}
                  aria-hidden="true"
                />
                <button
                  type="button"
                  onClick={() => selecionarModo('login')}
                  aria-pressed={!isRegister}
                  className={`relative z-10 inline-flex h-10 items-center justify-center gap-2 text-sm font-semibold transition-colors duration-300 ${
                    isRegister ? 'text-neutral-600 hover:text-neutral-950' : 'text-white'
                  }`}
                >
                  <LogIn size={15} />
                  Login
                </button>
                <button
                  type="button"
                  onClick={() => selecionarModo('register')}
                  aria-pressed={isRegister}
                  className={`relative z-10 inline-flex h-10 items-center justify-center gap-2 text-sm font-semibold transition-colors duration-300 ${
                    isRegister ? 'text-white' : 'text-neutral-600 hover:text-neutral-950'
                  }`}
                >
                  <UserPlus size={15} />
                  Cadastro
                </button>
              </div>
            </div>

            {googleClientId && (
              <div className="mb-7 space-y-5">
              
                  <div ref={googleButtonRef} className="auth-google-button min-h-11 w-full" />
                <div className="flex items-center gap-4">
                  <span className="h-px flex-1 bg-neutral-200" />
                  <span className="text-xs font-semibold uppercase tracking-[0.2em] text-neutral-500">ou</span>
                  <span className="h-px flex-1 bg-neutral-200" />
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              <div
                className={`grid transition-[grid-template-rows,opacity,transform] duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] ${
                  isRegister ? 'grid-rows-[1fr] opacity-100' : '-translate-y-2 grid-rows-[0fr] opacity-0'
                }`}
              >
                <div className="overflow-hidden">
                  <AuthInput
                    id="auth-nome"
                    label="Nome"
                    value={nome}
                    onChange={setNome}
                    Icon={User}
                    autoComplete="name"
                    minLength={2}
                    maxLength={160}
                    required={isRegister}
                  />
                </div>
              </div>

              <AuthInput
                id="auth-email"
                label="Email"
                value={email}
                onChange={setEmail}
                Icon={Mail}
                type="email"
                autoComplete="email"
                maxLength={255}
                required
              />

              <AuthInput
                id="auth-password"
                label="Senha"
                value={password}
                onChange={setPassword}
                Icon={Lock}
                type="password"
                autoComplete={isRegister ? 'new-password' : 'current-password'}
                minLength={8}
                maxLength={100}
                required
              />

              {erro && (
                <div className="border border-neutral-300 bg-neutral-100 px-3 py-2 text-sm font-medium text-neutral-900">
                  {erro}
                </div>
              )}

              <button
                type="submit"
                disabled={loading || googleLoading}
                className="group relative inline-flex h-[52px] w-full items-center justify-center overflow-hidden border border-neutral-950 bg-neutral-950 px-5 text-sm font-semibold uppercase tracking-[0.16em] text-white transition-colors duration-300 hover:text-neutral-950 disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:text-white"
              >
                <span className="absolute inset-0 translate-y-full bg-white transition-transform duration-300 ease-out group-hover:translate-y-0" />
                <span className="relative flex items-center gap-3">
                  {loading ? <Loader2 size={18} className="animate-spin" /> : <ArrowRight size={18} />}
                  {copy.action}
                </span>
              </button>
            </form>
          </div>
        </div>
      </section>
    </main>
  );
}
