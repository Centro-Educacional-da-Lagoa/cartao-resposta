"""
🌐 API REST Local - Sistema de Correção de Cartões
Rode localmente enquanto desenvolve, depois migre para Docker
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import tempfile
from datetime import datetime

# Importar funções do bot
from script import (
    configurar_google_sheets,
)

app = Flask(__name__)
CORS(app)  # Permitir acesso do React

GOOGLE_SHEETS_9ANO = os.getenv("GOOGLE_SHEETS_9ANO")
GOOGLE_SHEETS_5ANO = os.getenv("GOOGLE_SHEETS_5ANO")

# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.route('/')
def home():
    """Página inicial"""
    return jsonify({
        "status": "online",
        "mensagem": "Bot de Correção de Cartões - API REST",
        "versao": "1.0.0",
        "endpoints": [
            "GET  /api/status",
            "GET  /api/estatisticas_9ano",
            "GET  /api/estatisticas_5ano",
            "GET  /api/estadisticas_gerais",
        ]
    })

@app.route('/api/status')
def status():
    """Status do sistema"""
    try:
        client = configurar_google_sheets()

        sheets_9ano = client.open_by_key(GOOGLE_SHEETS_9ANO)
        sheets_5ano = client.open_by_key(GOOGLE_SHEETS_5ANO)

        dados9ano = sheets_9ano.get_all_records()
        dados5ano = sheets_5ano.get_all_records()

        data_9ano = None
        if dados9ano:
            data_9ano = dados9ano[-1].get('DATA')

        data_5ano = None
        if dados5ano:
            data_5ano = dados5ano[-1].get('DATA')

        ultima_atualizacao = None
        if data_9ano and data_5ano:
            ultima_atualizacao = max(data_9ano, data_5ano)
        elif data_9ano:
            ultima_atualizacao = data_9ano or data_5ano

        return jsonify({
            "status": "Em andamento",
            "timestamp": datetime.now().isoformat(),
            "bot_ativo": True,
            "ultima_atualizacao": ultima_atualizacao,
            "total_registros_9ano": len(dados9ano),
            "total_registros_5ano": len(dados5ano),
        })
    except Exception as e:
        print(f"Erro no endpoint {e}")
        return jsonify({
            "status": "error",
            "erro": str(e)
        }), 500

@app.route('/api/aluno/9ano')
def listar_alunos_9ano():
    """Listar alunos do 9° ano"""
    try:
        client = configurar_google_sheets()
        sheet = client.open_by_key(GOOGLE_SHEETS_9ANO).sheet1
        dados = sheet.get_all_records()

        return jsonify({
            "status": "success",
            "ano": "9º Ano",
            "total_alunos": len(dados),
            "alunos": dados,
        })
    except Exception as e:
        print(f"Erro no endpoint {e}")
        return jsonify({
            "status": "error",
            "erro": str(e)
        }), 500
    
@app.route('/api/aluno/5ano')
def listar_alunos_5ano():
    """Listar alunos do 5° ano"""
    try:
        client = configurar_google_sheets()
        sheet = client.open_by_key(GOOGLE_SHEETS_5ANO).sheet1
        dados = sheet.get_all_records()

        return jsonify({
            "status": "success",
            "ano": "5º Ano",
            "total_alunos": len(dados),
            "alunos": dados,
        })
    except Exception as e:
        print(f"Erro no endpoint {e}")
        return jsonify({
            "status": "error",
            "erro": str(e)
        }), 500
    
@app.route('/api/estatisticas/9ano')
def estatisticas_9ano():
    """Estatísticas do 9° ano"""
    try:
        client = configurar_google_sheets()
        sheet = client.open_by_key(GOOGLE_SHEETS_9ANO).sheet1
        dados = sheet.get_all_records()
        
        if not dados:
            return jsonify({
                "status": "success",
                "ano": "9°",
                "total_alunos": 0,
                "mensagem": "Nenhum dado encontrado"
            })
        
        # Calcular estatísticas
        percentuais = [float(d['Porcentagem'].replace('%', '')) for d in dados if d.get('Porcentagem')]
        
        return jsonify({
            "status": "success",
            "ano": "9°",
            "total_alunos": len(dados),
            "media_geral": sum(percentuais) / len(percentuais) if percentuais else 0,
            "nota_mais_alta": max(percentuais) if percentuais else 0,
            "nota_mais_baixa": min(percentuais) if percentuais else 0,
            "aprovados": sum(1 for p in percentuais if p >= 70),
            "reprovados": sum(1 for p in percentuais if p < 70)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "erro": str(e)
        }), 500

@app.route('/api/estatisticas/5ano')
def estatisticas_5ano():
    """Estatísticas do 5° ano"""
    try:
        client = configurar_google_sheets()
        sheet = client.open_by_key(GOOGLE_SHEETS_5ANO).sheet1
        dados = sheet.get_all_records()
        
        if not dados:
            return jsonify({
                "status": "success",
                "ano": "5°",
                "total_alunos": 0,
                "mensagem": "Nenhum dado encontrado"
            })
        
        # Calcular estatísticas
        percentuais = [float(d['Porcentagem'].replace('%', '')) for d in dados if d.get('Porcentagem')]
        
        return jsonify({
            "status": "success",
            "ano": "5°",
            "total_alunos": len(dados),
            "media_geral": sum(percentuais) / len(percentuais) if percentuais else 0,
            "nota_mais_alta": max(percentuais) if percentuais else 0,
            "nota_mais_baixa": min(percentuais) if percentuais else 0,
            "aprovados": sum(1 for p in percentuais if p >= 70),
            "reprovados": sum(1 for p in percentuais if p < 70)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "erro": str(e)
        }), 500

@app.route('/api/estatisticas/geral')
def estatisticas_geral():
    """Estatísticas consolidadas (ambos os anos)"""
    try:
        client = configurar_google_sheets()
        
        # Buscar dados de ambas as planilhas
        sheet_9ano = client.open_by_key(GOOGLE_SHEETS_9ANO).sheet1
        sheet_5ano = client.open_by_key(GOOGLE_SHEETS_5ANO).sheet1
        
        dados_9ano = sheet_9ano.get_all_records()
        dados_5ano = sheet_5ano.get_all_records()
        
        total_alunos = len(dados_9ano) + len(dados_5ano)
        
        # Calcular médias
        percentuais_9ano = [float(d['Porcentagem'].replace('%', '')) for d in dados_9ano if d.get('Porcentagem')]
        percentuais_5ano = [float(d['Porcentagem'].replace('%', '')) for d in dados_5ano if d.get('Porcentagem')]
        
        todos_percentuais = percentuais_9ano + percentuais_5ano
        
        return jsonify({
            "status": "success",
            "total_alunos": total_alunos,
            "media_geral": sum(todos_percentuais) / len(todos_percentuais) if todos_percentuais else 0,
            "por_ano": {
                "9ano": {
                    "total": len(dados_9ano),
                    "media": sum(percentuais_9ano) / len(percentuais_9ano) if percentuais_9ano else 0
                },
                "5ano": {
                    "total": len(dados_5ano),
                    "media": sum(percentuais_5ano) / len(percentuais_5ano) if percentuais_5ano else 0
                }
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "erro": str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 80)
    print("🌐 API REST - MODO LEITURA (Somente GET)")
    print("=" * 80)
    print("📊 Servidor iniciado em: http://localhost:5000")
    print("🔗 Endpoints disponíveis:")
    print("   - http://localhost:5000/api/status")
    print("   - http://localhost:5000/api/alunos/9ano")
    print("   - http://localhost:5000/api/alunos/5ano")
    print("   - http://localhost:5000/api/estatisticas/9ano")
    print("   - http://localhost:5000/api/estatisticas/5ano")
    print("   - http://localhost:5000/api/estatisticas/geral")
    print("=" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=True)