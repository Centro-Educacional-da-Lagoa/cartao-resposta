#!/usr/bin/env python3
"""
Monitor Automático de Cartões-Resposta
Verifica continuamente por novos arquivos e processa automaticamente
"""

import os
import time
import logging
import json
from datetime import datetime
from typing import Set, List, Dict
import subprocess
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor_automatico.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class MonitorAutomatico:
    def __init__(self):
        self.drive_folder_id = os.getenv('DRIVE_FOLDER_ID')
        self.intervalo_verificacao = 30  # segundos
        self.arquivos_processados = self.carregar_historico()
        self.contador_verificacoes = 0
        self.max_verificacoes = None      # ← NOVO: limite de verificações
        self.tempo_limite = None          # ← NOVO: tempo limite em horas
        
    def carregar_historico(self) -> Set[str]:
        """Carrega lista de arquivos já processados"""
        try:
            if os.path.exists('historico_processados.json'):
                with open('historico_processados.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(data.get('arquivos_processados', []))
        except Exception as e:
            logging.warning(f"Erro ao carregar histórico: {e}")
        return set()
    
    def salvar_historico(self, novos_arquivos: List[str] = None):
        """Salva histórico de arquivos processados"""
        try:
            if novos_arquivos:
                self.arquivos_processados.update(novos_arquivos)
            
            data = {
                'ultima_verificacao': datetime.now().isoformat(),
                'total_verificacoes': self.contador_verificacoes,
                'arquivos_processados': list(self.arquivos_processados)
            }
            with open('historico_processados.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Erro ao salvar histórico: {e}")
    
    def obter_arquivos_drive(self) -> List[Dict]:
        """Obtém lista de arquivos do Google Drive usando o script principal"""
        try:
            # Importar funções do script principal
            import script
            
            # Configurar Google Drive
            drive_service = script.configurar_google_drive_service_completo()
            if not drive_service:
                logging.error("Erro ao configurar Google Drive")
                return []
            
            # Listar arquivos
            query = f"'{self.drive_folder_id}' in parents and trashed = false"
            results = drive_service.files().list(
                q=query,
                fields="files(id, name, mimeType, modifiedTime, parents)",
                pageSize=100
            ).execute()
            
            arquivos = results.get('files', [])
            return arquivos
            
        except Exception as e:
            logging.error(f"Erro ao obter arquivos: {e}")
            return []
    
    def identificar_novos_cartoes(self, arquivos: List[Dict]) -> List[Dict]:
        """Identifica cartões novos que precisam ser processados"""
        novos_cartoes = []
        
        for arquivo in arquivos:
            arquivo_id = arquivo['id']
            nome = arquivo['name'].lower()
            
            # Verificar se é um cartão válido e novo
            if (arquivo_id not in self.arquivos_processados and
                'gabarito' not in nome and
                any(ext in nome for ext in ['.pdf', '.png', '.jpg', '.jpeg'])):
                
                novos_cartoes.append({
                    'id': arquivo_id,
                    'nome': arquivo['name'],
                    'modificado': arquivo.get('modifiedTime', 'N/A')
                })
        
        return novos_cartoes
    
    def processar_cartoes(self, novos_cartoes: List[Dict]) -> bool:
        """Executa o processamento dos cartões usando o script principal"""
        try:
            logging.info(f"Iniciando processamento de {len(novos_cartoes)} cartões...")
            
            # Listar cartões que serão processados
            for cartao in novos_cartoes:
                logging.info(f"  -> {cartao['nome']}")
            
            # Executar script principal
            resultado = subprocess.run([
                'python', 'script.py',
                '--drive-folder', self.drive_folder_id
            ], capture_output=True, text=True, timeout=600)  # 10 minutos timeout
            
            if resultado.returncode == 0:
                logging.info("Processamento concluído com sucesso!")
                
                # Marcar cartões como processados
                ids_processados = [cartao['id'] for cartao in novos_cartoes]
                self.salvar_historico(ids_processados)
                
                # Log das últimas linhas da saída
                if resultado.stdout:
                    linhas_saida = resultado.stdout.strip().split('\n')
                    for linha in linhas_saida[-5:]:  # Últimas 5 linhas
                        if linha.strip():
                            logging.info(f"SAIDA: {linha.strip()}")
                
                return True
            else:
                logging.error(f"Erro no processamento (código {resultado.returncode})")
                if resultado.stderr:
                    # Log das últimas linhas do erro
                    linhas_erro = resultado.stderr.strip().split('\n')
                    for linha in linhas_erro[-3:]:  # Últimas 3 linhas
                        if linha.strip():
                            logging.error(f"ERRO: {linha.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error("TIMEOUT: Processamento demorou mais de 10 minutos")
            return False
        except Exception as e:
            logging.error(f"Erro no processamento: {e}")
            return False
    
    def verificar_e_processar(self):
        """Executa verificação e processamento se necessário"""
        try:
            self.contador_verificacoes += 1
            logging.info(f"=== VERIFICACAO #{self.contador_verificacoes} - {datetime.now().strftime('%H:%M:%S')} ===")
            
            # Obter arquivos do Drive
            arquivos = self.obter_arquivos_drive()
            if not arquivos:
                logging.warning("Nenhum arquivo encontrado na pasta")
                return
            
            logging.info(f"Total de arquivos na pasta: {len(arquivos)}")
            
            # Identificar novos cartões
            novos_cartoes = self.identificar_novos_cartoes(arquivos)
            
            if novos_cartoes:
                logging.info(f">>> ENCONTRADOS {len(novos_cartoes)} NOVOS CARTOES <<<")
                
                # Processar cartões
                sucesso = self.processar_cartoes(novos_cartoes)
                
                if sucesso:
                    logging.info(">>> PROCESSAMENTO CONCLUIDO COM SUCESSO <<<")
                    logging.info("Cartões movidos para pasta 'cartoes-processados'")
                else:
                    logging.error(">>> FALHA NO PROCESSAMENTO <<<")
            else:
                logging.info("Nenhum cartão novo encontrado")
                # Atualizar histórico com arquivos atuais para evitar reprocessamento
                ids_atuais = [arquivo['id'] for arquivo in arquivos]
                known_files = len([id for id in ids_atuais if id in self.arquivos_processados])
                logging.info(f"Arquivos já conhecidos: {known_files}/{len(arquivos)}")
            
        except Exception as e:
            logging.error(f"Erro na verificação: {e}")
        finally:
            logging.info("=" * 60)
    
    def iniciar_monitoramento(self, intervalo_minutos: int = 5):
        """Inicia monitoramento contínuo"""
        intervalo_segundos = intervalo_minutos * 60
        
        logging.info("=" * 60)
        logging.info("MONITOR AUTOMATICO DE CARTOES-RESPOSTA INICIADO")
        logging.info(f"Pasta Drive: {self.drive_folder_id}")
        logging.info(f"Intervalo: {intervalo_minutos} minutos ({intervalo_segundos}s)")
        logging.info(f"Arquivos já processados: {len(self.arquivos_processados)}")
        logging.info("Pressione Ctrl+C para parar")
        logging.info("=" * 60)
        
        try:
            # Primeira verificação imediata
            self.verificar_e_processar()
            
            # Loop principal
            while True:
                time.sleep(intervalo_segundos)
                self.verificar_e_processar()
                
        except KeyboardInterrupt:
            logging.info("MONITOR INTERROMPIDO PELO USUARIO")
            logging.info(f"Total de verificações realizadas: {self.contador_verificacoes}")
            self.salvar_historico()
        except Exception as e:
            logging.error(f"ERRO CRITICO NO MONITOR: {e}")
            self.salvar_historico()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor automático de cartões-resposta"
    )
    parser.add_argument(
        '--intervalo',
        type=int,
        default=5,
        help='Intervalo de verificação em minutos (padrão: 5)'
    )
    parser.add_argument(
        '--testar',
        action='store_true',
        help='Executa uma verificação única para teste'
    )
    
    args = parser.parse_args()
    
    monitor = MonitorAutomatico()
    
    if args.testar:
        logging.info("=== MODO TESTE ===")
        monitor.verificar_e_processar()
    else:
        monitor.iniciar_monitoramento(args.intervalo)

if __name__ == "__main__":
    main()