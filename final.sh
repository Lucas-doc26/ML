set -e

#Variáveis
AUTOENCODERS=("Kyoto" "CNR" "PKLot")
CLASSIFIERS_KYOTO=("PUC" "UFPR05" "UFPR04" "camera1" "camera2" "camera3" "camera4" "camera5" "camera6" "camera7" "camera8" "camera9")
CLASSIFIERS_CNR=("PUC" "UFPR04" "UFPR05")
CLASSIFIERS_PKLOT=("camera1" "camera2" "camera3" "camera4" "camera5" "camera6" "camera7" "camera8" "camera9")
FACULDADES=("PUC" "UFPR04" "UFPR05")

NAME_MODEL="Modelo_Kyoto"

#Organizar datasets
#python3 datasets.py

#Treino autoencoders
#for AE in "${AUTOENCODERS[@]}"; do
    #EPOCHS=50
    #if [ "$AE" == "Kyoto" ]; then
     #   EPOCHS=200
    #fi
    #python3 train_autoencoders.py --name_model "$NAME_MODEL" --autoencoder_base "$AE" --autoencoder_epochs "$EPOCHS"
#done

#Treino classificadores
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

#Testando na Kyoto
for CL in "${CLASSIFIERS_KYOTO[@]}"; do
    TEST_BASES=()
    for item in "${CLASSIFIERS_KYOTO[@]}"; do
        if [[ "$item" != "$CL" ]]; then
            TEST_BASES+=("$item")
        fi
    done

    python3 test_classifiers.py \
        --name_model "$NAME_MODEL" \
        --autoencoder_base "Kyoto" \
        --classifier_base "$CL" \
        --test_bases "${TEST_BASES[@]}"
done

#Testando no CNR
for CL in "${CLASSIFIERS_KYOTO[@]}"; do
    TEST_BASES=()
    for item in "${CLASSIFIERS_KYOTO[@]}"; do
        if [[ "$item" != "$CL" ]]; then
            TEST_BASES+=("$item")
        fi
    done

    python3 test_classifiers.py \
        --name_model "$NAME_MODEL" \
        --autoencoder_base "CNR" \
        --classifier_base "$CL" \
        --test_bases "${TEST_BASES[@]}"
done

#Testando na PKLot
for CL in "${CLASSIFIERS_KYOTO[@]}"; do
    TEST_BASES=()
    for item in "${CLASSIFIERS_KYOTO[@]}"; do
        if [[ "$item" != "$CL" ]]; then
            TEST_BASES+=("$item")
        fi
    done

    python3 test_classifiers.py \
        --name_model "$NAME_MODEL" \
        --autoencoder_base "PKLot" \
        --classifier_base "$CL" \
        --test_bases "${TEST_BASES[@]}"
done

#Resultados
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

#Fusões 
python3 teste_fusoes.py

#Gerando tabela 
python tabela.py 