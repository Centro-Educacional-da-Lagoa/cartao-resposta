import os
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from storage_vultr import VultrS3Config, VultrS3Storage


ENV_S3 = {
    "VULTR_S3_ACCESS_KEY_ID": "access-key",
    "VULTR_S3_SECRET_ACCESS_KEY": "secret-key",
    "VULTR_S3_HOST": "ewr1.vultrobjects.com",
    "VULTR_S3_BUCKET": "cartoes",
}


class VultrS3ConfigTest(unittest.TestCase):
    def test_carrega_configuracao_e_normaliza_prefixos(self):
        with patch.dict(os.environ, ENV_S3, clear=True):
            config = VultrS3Config.from_env()

        self.assertEqual(config.endpoint_url, "https://ewr1.vultrobjects.com")
        self.assertEqual(config.region, "ewr1")
        self.assertEqual(config.prefixo_upload, "entrada/")
        self.assertEqual(config.prefixo_gabaritos, "gabaritos/")
        self.assertEqual(config.prefixo_processados, "processados/")


class VultrS3StorageTest(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.list_objects_v2.side_effect = [
            {
                "Contents": [
                    {
                        "Key": "entrada/id-1/turma-a.pdf",
                        "LastModified": datetime(2026, 6, 11, tzinfo=timezone.utc),
                        "Size": 123,
                        "ETag": '"etag-1"',
                    }
                ],
                "IsTruncated": True,
                "NextContinuationToken": "pagina-2",
            },
            {
                "Contents": [
                    {
                        "Key": "entrada/id-2/turma-b.pdf",
                        "LastModified": datetime(2026, 6, 11, tzinfo=timezone.utc),
                        "Size": 456,
                        "ETag": '"etag-2"',
                    }
                ],
                "IsTruncated": False,
            },
        ]

        with patch.dict(os.environ, ENV_S3, clear=True), patch(
            "storage_vultr.boto3.client",
            return_value=self.client,
        ):
            self.storage = VultrS3Storage.from_env()

    def test_lista_uploads_com_paginacao(self):
        arquivos = self.storage.listar_uploads()

        self.assertEqual([arquivo["name"] for arquivo in arquivos], ["turma-a.pdf", "turma-b.pdf"])
        self.assertEqual(arquivos[0]["id"], "s3:entrada/id-1/turma-a.pdf")
        self.assertEqual(arquivos[0]["storage_id"], "entrada/id-1/turma-a.pdf")
        self.assertEqual(arquivos[0]["source"], "vultr_s3")
        self.assertEqual(
            self.client.list_objects_v2.call_args_list[1].kwargs["ContinuationToken"],
            "pagina-2",
        )

    def test_move_objeto_para_prefixo_do_ano(self):
        destino = self.storage.mover_para_processados(
            "entrada/id-1/turma-a.pdf",
            "5ano",
        )

        self.assertEqual(destino, "processados/5ano/id-1/turma-a.pdf")
        self.client.copy_object.assert_called_once_with(
            Bucket="cartoes",
            CopySource={
                "Bucket": "cartoes",
                "Key": "entrada/id-1/turma-a.pdf",
            },
            Key="processados/5ano/id-1/turma-a.pdf",
            ContentType="application/pdf",
            MetadataDirective="REPLACE",
        )
        self.client.delete_object.assert_called_once_with(
            Bucket="cartoes",
            Key="entrada/id-1/turma-a.pdf",
        )


if __name__ == "__main__":
    unittest.main()
