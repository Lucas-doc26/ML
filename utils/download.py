import tarfile
import urllib.request
from pathlib import Path
import requests
import zipfile
import shutil
import os
from .preprocessing import data_augmentation_kyoto

def download_Kyoto(path):
    """
    Função para baixar e extrair o dataset Kyoto
    """
    # Criar pasta Kyoto dentro do caminho especificado
    kyoto_path = Path(path) / "Kyoto"
    kyoto_path.mkdir(exist_ok=True)
    print(f"Pasta Kyoto criada em: {kyoto_path.absolute()}")
    
    # Baixar o arquivo ZIP
    url = "https://github.com/eizaburo-doi/kyoto_natim/archive/refs/heads/master.zip"
    print("Baixando arquivo ZIP...")
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Erro ao baixar o arquivo. Status code: {response.status_code}")
    
    # Salvar o arquivo ZIP temporariamente no caminho especificado
    zip_path = Path(path) / "kyoto_temp.zip"
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    print(f"Arquivo ZIP salvo em: {zip_path}")
    
    # Extrair o arquivo ZIP
    print("Extraindo arquivo ZIP...")
    temp_extract_path = Path(path) / "temp_extract"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)
    
    # Listar todos os arquivos extraídos para debug
    print("\nConteúdo extraído:")
    for root, dirs, files in os.walk(temp_extract_path):
        print(f"\nDiretório: {root}")
        for name in files:
            print(f"- {name}")
    
    # Tentar diferentes possíveis caminhos para a pasta thumb
    possible_paths = [
        temp_extract_path / "kyoto_natim-master/kyoto_natim-master/thumb",
        temp_extract_path / "kyoto_natim-master/thumb",
        temp_extract_path / "thumb"
    ]
    
    thumb_path = None
    for p in possible_paths:
        if p.exists():
            thumb_path = p
            print(f"\nPasta thumb encontrada em: {thumb_path}")
            break
    
    if not thumb_path:
        raise Exception("Pasta thumb não encontrada nos caminhos esperados")
    
    # Copiar todas as imagens para a pasta Kyoto
    print("\nCopiando imagens...")
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    images_copied = 0
    
    for file in thumb_path.glob("*"):
        print(f"Encontrado arquivo: {file.name}")
        if file.suffix.lower() in image_extensions:
            shutil.copy2(file, kyoto_path)
            print(f"Copiado: {file.name}")
            images_copied += 1
    
    # Limpar arquivos temporários
    print("\nLimpando arquivos temporários...")
    os.remove(zip_path)
    shutil.rmtree(temp_extract_path)
    
    if images_copied == 0:
        print("\nNenhuma imagem foi encontrada para copiar!")
    else:
        print(f"\nProcesso concluído! {images_copied} imagens foram copiadas para a pasta Kyoto")

def download_datasets(path, link_datasets):
    """
    Função para baixar e extrair um dataset.tar.gz 
    """
    name = link_datasets.split('/')[-1]
    file_name = Path(path) / name

    print(f"Baixando arquivo {file_name}...")
    urllib.request.urlretrieve(link_datasets, file_name)

    print(f"Extraindo arquivo {file_name}...")
    with tarfile.open(file_name, 'r:gz') as tar:
        tar.extractall(path=path)

    print(f"Arquivo {file_name} extraído com sucesso!")

    return file_name

def download_all_datasets(path_datasets):
    """
    Função para baixar e extrair todos os datasets
    """
    datasets = os.listdir(path_datasets)
    if 'PKLot' not in datasets:
        download_datasets(path_datasets, 'http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz')
    if 'Kyoto' not in datasets:
        download_Kyoto(path_datasets)
    if 'CNR-EXT-Patches-150x150' not in datasets:
        download_datasets(path_datasets, 'https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip')
    else:
        print("Todos os datasets já no sistema!")

    return os.path.join(path_datasets , 'PKLot', 'PKLotSegmented'), os.path.join(path_datasets , 'CNR-EXT-Patches-150x150'), os.path.join(path_datasets , 'Kyoto')
