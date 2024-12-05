import shutil
import os
import time

pasta_local = "/home/lucas/PIBIC (copy)/Modelos"

destino_drive = "/run/user/1005/gvfs/google-drive:host=gmail.com,user=lucaas.ocunha/0AFbM129_bq40Uk9PVA/Modelos"

time.sleep(1)

try:
    # Verifica se a pasta de destino já existe
    if os.path.exists(destino_drive):
        print(f"Erro: A pasta de destino {destino_drive} já existe.")
    else:
        # Copia a pasta para o Google Drive
        shutil.copytree(pasta_local, destino_drive)
        print(f"Pasta {pasta_local} copiada com sucesso para {destino_drive}!")
except Exception as e:
    print(f"Erro ao copiar a pasta: {e}")
