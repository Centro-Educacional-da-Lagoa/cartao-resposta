import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from storage_google_drive import GoogleDriveConfig, GoogleDriveStorage


ENV_DRIVE = {
    "DRIVER_FOLDER_ID": "entrada",
    "DRIVER_FOLDER_4ANO": "destino-4",
    "DRIVER_FOLDER_5ANO": "destino-5",
    "DRIVER_FOLDER_8ANO": "destino-8",
    "DRIVER_FOLDER_9ANO": "destino-9",
}


class GoogleDriveConfigTest(unittest.TestCase):
    def test_carrega_pastas_por_ano(self):
        with patch.dict(os.environ, ENV_DRIVE, clear=True):
            config = GoogleDriveConfig.from_env()

        self.assertEqual(config.pasta_upload_id, "entrada")
        self.assertEqual(config.pastas_processados["5ano"], "destino-5")


class GoogleDriveStorageTest(unittest.TestCase):
    def setUp(self):
        self.service = MagicMock()
        self.storage = GoogleDriveStorage(
            GoogleDriveConfig(
                pasta_upload_id="entrada",
                pastas_processados={
                    "4ano": "destino-4",
                    "5ano": "destino-5",
                    "8ano": "destino-8",
                    "9ano": "destino-9",
                },
            ),
            self.service,
        )

    def test_lista_uploads_com_id_da_origem(self):
        self.service.files.return_value.list.return_value.execute.return_value = {
            "files": [
                {
                    "id": "arquivo-1",
                    "name": "turma.pdf",
                    "mimeType": "application/pdf",
                    "modifiedTime": "2026-06-11T10:00:00Z",
                    "size": "123",
                }
            ]
        }

        arquivos = self.storage.listar_uploads()

        self.assertEqual(arquivos[0]["id"], "drive:arquivo-1")
        self.assertEqual(arquivos[0]["storage_id"], "arquivo-1")
        self.assertEqual(arquivos[0]["source"], "google_drive")

    def test_move_para_pasta_do_ano(self):
        self.service.files.return_value.get.return_value.execute.return_value = {
            "parents": ["entrada"]
        }

        destino = self.storage.mover_para_processados("arquivo-1", "5ano")

        self.assertEqual(destino, "destino-5")
        self.service.files.return_value.update.assert_called_once_with(
            fileId="arquivo-1",
            addParents="destino-5",
            removeParents="entrada",
            fields="id, parents",
            supportsAllDrives=True,
        )

if __name__ == "__main__":
    unittest.main()
