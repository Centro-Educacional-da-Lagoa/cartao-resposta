import { useRef, useState, type ChangeEvent } from 'react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import * as XLSX from 'xlsx';
import { api, type Aluno, type AnoFiltro } from '../service/api';
import { FileText, FileSpreadsheet, Upload } from 'lucide-react';

const TIPO_PDF = 'application/pdf';

type UploadFeedback = {
    tipo: 'sucesso' | 'erro';
    mensagem: string;
};

interface Props {
    alunos: Aluno[];
    ano: AnoFiltro;
}

export function BotoesExportacao({ alunos, ano }: Props) {
    const inputArquivoRef = useRef<HTMLInputElement | null>(null);
    const [enviandoArquivo, setEnviandoArquivo] = useState(false);
    const [feedbackUpload, setFeedbackUpload] = useState<UploadFeedback | null>(null);

    const exportarParaPDF = () => {
        const doc = new jsPDF() as jsPDF & { internal: { getNumberOfPages: () => number } };

        const DadosOrdenados = [...alunos].sort((a, b) => {
            const dataA = new Date(a.DATA.split('/').reverse().join('-'));
            const dataB = new Date(b.DATA.split('/').reverse().join('-'));
            return dataB.getTime() - dataA.getTime();
        })

        doc.setFontSize(18);
        doc.setTextColor(30, 30, 30);
        doc.text(`Relatório de Alunos - ${ano === 'geral' ? 'Todos os Anos' : ano + 'º Ano'}`, 14, 20);

        doc.setFontSize(11);
        doc.setTextColor(100, 100, 100);
        doc.text(`Centro Educacional da Lagoa`, 14, 28);
        doc.text(`Data: ${new Date().toLocaleDateString('pt-BR')}`, 14, 34);
        doc.text(`Total de alunos: ${alunos.length}`, 14, 40);

        doc.setDrawColor(200, 200, 200);
        doc.line(14, 44, 196, 44);

        const dadosTabela = DadosOrdenados.map(aluno => [
            aluno.DATA,
            aluno["Nome completo"],
            aluno.Escola,
            aluno.Turma,
            aluno.Porcentagem,
            aluno["Acertos Língua Portuguesa"],
            aluno["Acertos Matemática"],
            aluno["Erros Língua Portuguesa"],
            aluno["Erros Matemática"]
        ]);

        autoTable(doc, {
            head: [['Data', 'Nome', 'Escola', 'Turma', 'Nota', 'Port.', 'Mat.', 'Erros Port.', 'Erros Mat.']],
            body: dadosTabela,
            startY: 48,
            styles: {
                fontSize: 8,
                cellPadding: 3
            },
            headStyles: {
                fillColor: [59, 130, 246],
                textColor: [255, 255, 255],
                fontStyle: 'bold'
            },
            alternateRowStyles: {
                fillColor: [245, 247, 250]
            },
            didParseCell: function (data) {
                if (data.column.index === 4 && data.section === 'body') {
                    const nota = parseFloat(data.cell.text[0].replace('%', '').replace(',', '.'));
                    if (nota >= 70) {
                        data.cell.styles.textColor = [16, 185, 129];
                        data.cell.styles.fontStyle = 'bold';
                    } else {
                        data.cell.styles.textColor = [239, 68, 68];
                        data.cell.styles.fontStyle = 'bold';
                    }
                }
            }
        });

        const pageCount = doc.internal.getNumberOfPages();
        doc.setFontSize(9);
        doc.setTextColor(150, 150, 150);
        for (let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.text(
                `Página ${i} de ${pageCount}`,
                doc.internal.pageSize.width / 2,
                doc.internal.pageSize.height - 10,
                { align: 'center' }
            );
        }

        doc.save(`relatorio-alunos-${ano}-${Date.now()}.pdf`);
    };

    const exportarParaExcel = () => {
        const dadosExcel = alunos.map(aluno => ({
            'Data': aluno.DATA,
            'Nome Completo': aluno["Nome completo"],
            'Escola': aluno.Escola,
            'Turma': aluno.Turma,
            'Porcentagem': aluno.Porcentagem,
            'Acertos Português': aluno["Acertos Língua Portuguesa"],
            'Erros Português': aluno["Erros Língua Portuguesa"],
            'Acertos Matemática': aluno["Acertos Matemática"],
            'Erros Matemática': aluno["Erros Matemática"]
        }));

        const ws = XLSX.utils.json_to_sheet(dadosExcel);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, ano === 'geral' ? 'Todos' : `${ano}º Ano`);

        const wscols = [
            { wch: 12 }, { wch: 30 }, { wch: 25 }, { wch: 10 }, { wch: 12 },
            { wch: 15 }, { wch: 15 }, { wch: 15 }, { wch: 15 }
        ];
        ws['!cols'] = wscols;

        XLSX.writeFile(wb, `relatorio-alunos-${ano}-${Date.now()}.xlsx`);
    };

    const exportarCSV = () => {
        const dadosCSV = alunos.map(aluno => ({
            'Data': aluno.DATA,
            'Nome Completo': aluno["Nome completo"],
            'Escola': aluno.Escola,
            'Turma': aluno.Turma,
            'Porcentagem': aluno.Porcentagem,
            'Acertos Português': aluno["Acertos Língua Portuguesa"],
            'Erros Português': aluno["Erros Língua Portuguesa"],
            'Acertos Matemática': aluno["Acertos Matemática"],
            'Erros Matemática': aluno["Erros Matemática"]
        }));

        const ws = XLSX.utils.json_to_sheet(dadosCSV);
        const csv = XLSX.utils.sheet_to_csv(ws);

        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `relatorio-alunos-${ano}-${Date.now()}.csv`;
        link.click();
    };

    const abrirSeletorArquivo = () => {
        inputArquivoRef.current?.click();
    };

    const arquivoValido = (arquivo: File) => {
        const extensao = arquivo.name.split('.').pop()?.toLowerCase();
        return extensao === 'pdf' && (arquivo.type === '' || arquivo.type === TIPO_PDF);
    };

    const enviarArquivo = async (event: ChangeEvent<HTMLInputElement>) => {
        const arquivo = event.target.files?.[0];
        if (!arquivo) {
            return;
        }

        setFeedbackUpload(null);

        if (!arquivoValido(arquivo)) {
            setFeedbackUpload({
                tipo: 'erro',
                mensagem: 'Formato inválido. Selecione um arquivo PDF.'
            });
            event.target.value = '';
            return;
        }

        try {
            setEnviandoArquivo(true);
            const resposta = await api.uploadArquivo(arquivo);
            setFeedbackUpload({
                tipo: 'sucesso',
                mensagem:
                    typeof resposta.message === 'string' && resposta.message.trim().length > 0
                        ? resposta.message
                        : `Arquivo ${arquivo.name} enviado com sucesso.`
            });
        } catch (error) {
            console.error('Erro ao enviar arquivo:', error);
            setFeedbackUpload({
                tipo: 'erro',
                mensagem: 'Falha ao enviar arquivo. Tente novamente em instantes.'
            });
        } finally {
            setEnviandoArquivo(false);
            event.target.value = '';
        }
    };

    return (
        <div className="bg-white rounded-xl p-5 shadow-sm mb-8">
            <input
                ref={inputArquivoRef}
                type="file"
                accept=".pdf,application/pdf"
                onChange={enviarArquivo}
                className="hidden"
            />

            <div className="flex gap-3 flex-wrap">
                <button
                    type="button"
                    onClick={exportarParaPDF}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg font-medium text-sm shadow-sm hover:shadow-md hover:-translate-y-0.5 active:translate-y-0 transition-all"
                >
                    <FileText size={18} />
                    Exportar PDF
                </button>
                <button
                    type="button"
                    onClick={exportarParaExcel}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg font-medium text-sm shadow-sm hover:shadow-md hover:-translate-y-0.5 active:translate-y-0 transition-all"
                >
                    <FileSpreadsheet size={18} />
                    Exportar Excel
                </button>
                <button
                    type="button"
                    onClick={exportarCSV}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg font-medium text-sm shadow-sm hover:shadow-md hover:-translate-y-0.5 active:translate-y-0 transition-all"
                >
                    <FileSpreadsheet size={18} />
                    Exportar CSV
                </button>

                <button
                    type="button"
                    onClick={abrirSeletorArquivo}
                    disabled={enviandoArquivo}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-amber-500 to-orange-600 text-white rounded-lg font-medium text-sm shadow-sm hover:shadow-md hover:-translate-y-0.5 active:translate-y-0 transition-all disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:translate-y-0"
                >
                    <Upload size={18} />
                    {enviandoArquivo ? 'Enviando PDF...' : 'Enviar PDF'}
                </button>
            </div>

            {feedbackUpload && (
                <p
                    className={`mt-3 text-sm font-medium ${
                        feedbackUpload.tipo === 'sucesso' ? 'text-emerald-600' : 'text-red-600'
                    }`}
                >
                    {feedbackUpload.mensagem}
                </p>
            )}
        </div>
    );
}
