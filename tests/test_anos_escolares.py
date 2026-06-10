import unittest

from anos_escolares import (
    detectar_ano_escolar,
    detectar_ano_por_turma,
    nome_gabarito,
    numero_questoes_por_ano,
)


class AnosEscolaresTest(unittest.TestCase):
    def test_detecta_codigos_e_rotulos(self):
        casos = {
            "4ano": "4ano",
            "Turma do 5º ano": "5ano",
            "oitavo ano do Ensino Fundamental": "8ano",
            "9° ano": "9ano",
            "Cartão do 4º": "4ano",
        }

        for texto, esperado in casos.items():
            with self.subTest(texto=texto):
                self.assertEqual(detectar_ano_escolar(texto), esperado)

    def test_mapeia_quantidade_de_questoes(self):
        self.assertEqual(numero_questoes_por_ano("4ano"), 44)
        self.assertEqual(numero_questoes_por_ano("5ano"), 44)
        self.assertEqual(numero_questoes_por_ano("8ano"), 52)
        self.assertEqual(numero_questoes_por_ano("9ano"), 52)

    def test_detecta_codigo_curto_da_turma(self):
        self.assertEqual(detectar_ano_por_turma("4A"), "4ano")
        self.assertEqual(detectar_ano_por_turma("Turma 8B"), "8ano")

    def test_monta_nome_padrao_do_gabarito(self):
        self.assertEqual(nome_gabarito("4ano"), "gabarito_4ano")
        self.assertEqual(nome_gabarito("8ano"), "gabarito_8ano")

    def test_rejeita_ano_nao_suportado(self):
        self.assertIsNone(detectar_ano_escolar("6º ano"))
        self.assertIsNone(numero_questoes_por_ano("N/A"))


if __name__ == "__main__":
    unittest.main()
