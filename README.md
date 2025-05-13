# PIBIC 2024-2025

## Configuração do Ambiente

### Requisitos do Sistema
- Python 3.10
- CUDA 11.8 (se usar GPU)
- 16GB RAM (mínimo)
- 50GB espaço em disco

### Instalação

1. **Instalar o Conda**:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

2. **Criar e ativar o ambiente**:
```bash
conda env create -f environment.yml
conda activate pibic-2024-2025
```

3. **Verificar a instalação**:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

4. **Dependências do Linux**:
```bash
export QT_QPA_PLATFORM=offscreen
```

### Configurações de Reproducibilidade

As configurações de reprodutibilidade estão centralizadas no arquivo `utils/config.py`. Para usar:

```python
from utils.config import set_seeds, config_gpu

# As configurações já serão aplicadas automaticamente ao importar
```

O arquivo configura:
- Seeds do TensorFlow, NumPy e Python
- Configurações determinísticas do TensorFlow
- Configurações da GPU
- Configurações do Pandas

### Estrutura de Diretórios
```
PIBIC-2024-2025/
├── datasets/
├── Modelos/
├── Pesos_parciais/
├── resultados/
└── utils/
    └── config.py  # Configurações globais
```

### Versões das Bibliotecas
- TensorFlow: 2.15.0
- NumPy: 1.24.3
- Pandas: 2.1.4
- Matplotlib: 3.8.2
- Scikit-learn: 1.2.2
- OpenCV: 4.8.1.78 