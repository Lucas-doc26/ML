#!/bin/bash
set -e

# Variáveis
AUTOENCODERS=("Kyoto" "CNR" "PKLot")
CLASSIFIERS_KYOTO=("PUC" "UFPR05" "UFPR04" "camera1" "camera2" "camera3" "camera4" "camera5" "camera6" "camera7" "camera8" "camera9")
CLASSIFIERS_CNR=("PUC" "UFPR04" "UFPR05")
CLASSIFIERS_PKLOT=("camera1" "camera2" "camera3" "camera4" "camera5" "camera6" "camera7" "camera8" "camera9")
FACULDADES=("PUC" "UFPR04" "UFPR05")

NAME_MODEL="Modelo_Kyoto"

# Organizar datasets
python3 datasets.py

#Gera os modelos 
python3 generate_autoencoders.py

# Treino autoencoders
for AE in "${AUTOENCODERS[@]}"; do
    EPOCHS=50
    if [ "$AE" == "Kyoto" ]; then
        EPOCHS=200
    fi
    python3 train_autoencoders.py --name_model "$NAME_MODEL" --autoencoder_base "$AE" --autoencoder_epochs "$EPOCHS"
done

# Treino classificadores
for AE in "${AUTOENCODERS[@]}"; do
    CLASSIFIERS=()
    if [ "$AE" == "Kyoto" ]; then
        CLASSIFIERS=("${CLASSIFIERS_KYOTO[@]}")
    elif [ "$AE" == "CNR" ]; then
        CLASSIFIERS=("${CLASSIFIERS_CNR[@]}")
    else
        CLASSIFIERS=("${CLASSIFIERS_PKLOT[@]}")
    fi

    for CL in "${CLASSIFIERS[@]}"; do
        python3 train_classifiers.py --name "$NAME_MODEL" --classifier_base "$CL" --classifier_epochs 20 --autoencoder_base "$AE"
    done
done

# Testes - exemplo apenas para Kyoto
for CL in "${CLASSIFIERS_KYOTO[@]}"; do
    TEST_BASES=("${CLASSIFIERS_KYOTO[@]/$CL}")  # remove o próprio da lista
    TEST_BASES+=("${FACULDADES[@]}")  # adiciona as faculdades

    python3 test_classifiers.py --name_model "$NAME_MODEL" --autoencoder_base "Kyoto" --classifier_base "$CL" --test_bases "${TEST_BASES[@]}"
done

# Testes para CNR
for CL in "${CLASSIFIERS_CNR[@]}"; do
    TEST_BASES=("${FACULDADES[@]/$CL}")  # remove o próprio
    python3 test_classifiers.py --name_model "$NAME_MODEL" --autoencoder_base "CNR" --classifier_base "$CL" --test_bases "${TEST_BASES[@]}"
done

# Testes para PKLot
for CL in "${CLASSIFIERS_PKLOT[@]}"; do
    TEST_BASES=("${CLASSIFIERS_PKLOT[@]/$CL}")  # remove o próprio
    python3 test_classifiers.py --name_model "$NAME_MODEL" --autoencoder_base "PKLot" --classifier_base "$CL" --test_bases "${TEST_BASES[@]}"
done

# Resultados
for AE in "${AUTOENCODERS[@]}"; do
    if [ "$AE" == "Kyoto" ]; then
        CLASSIFIERS=("${CLASSIFIERS_KYOTO[@]}" "${FACULDADES[@]}")
    elif [ "$AE" == "CNR" ]; then
        CLASSIFIERS=("${CLASSIFIERS_CNR[@]}")
    else
        CLASSIFIERS=("${CLASSIFIERS_PKLOT[@]}")
    fi

    python3 mean_results.py --name_model "$NAME_MODEL" --autoencoder_base "$AE"
done
