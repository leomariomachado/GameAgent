#!/usr/bin/env bash
set -Eeuo pipefail

# Baixa ImageNet/COCO pelo Kaggle, organiza os arquivos, compila o BaseAI
# e gera os shards.
#
# Uso:
#   ./rebuild_datasets.sh coco
#   ./rebuild_datasets.sh imagenet
#   ./rebuild_datasets.sh all
#
# Variáveis opcionais:
#   WORKSPACE=/workspace
#   BASEAI_DIR=/tmp/BaseAI
#   BUILD_DIR=/tmp/BaseAI/buildrelease
#   JOBS=32
#   KEEP_DOWNLOADS=0
#   FORCE_SHARDS=0

MODE="${1:-all}"

WORKSPACE="${WORKSPACE:-/workspace}"
BASEAI_DIR="${BASEAI_DIR:-/tmp/BaseAI}"
BUILD_DIR="${BUILD_DIR:-${BASEAI_DIR}/buildrelease}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${WORKSPACE}/downloads}"
JOBS="${JOBS:-$(nproc)}"
KEEP_DOWNLOADS="${KEEP_DOWNLOADS:-0}"
FORCE_SHARDS="${FORCE_SHARDS:-0}"

IMAGENET_DIR="${IMAGENET_DIR:-${WORKSPACE}/imagenet}"
IMAGENET_SHARDS="${IMAGENET_SHARDS:-${WORKSPACE}/imagenet_shards}"
COCO_DIR="${COCO_DIR:-${WORKSPACE}/coco}"
COCO_SHARDS="${COCO_SHARDS:-${WORKSPACE}/coco_shards}"

IMAGENET_COMPETITION="${IMAGENET_COMPETITION:-imagenet-object-localization-challenge}"
COCO_DATASET="${COCO_DATASET:-awsaf49/coco-2017-dataset}"

log() {
    printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date '+%F %T')" "$*"
}

die() {
    printf '\n\033[1;31mERRO:\033[0m %s\n' "$*" >&2
    exit 1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

install_dependencies() {
    log "Verificando dependências"

    if ! command_exists python3; then
        die "python3 não está instalado."
    fi

    if ! command_exists unzip; then
        if command_exists apt-get; then
            apt-get update
            DEBIAN_FRONTEND=noninteractive apt-get install -y unzip
        else
            die "Instale o comando unzip."
        fi
    fi

    if ! command_exists cmake; then
        die "cmake não está instalado."
    fi

    if ! command_exists kaggle; then
        log "Instalando Kaggle CLI"
        python3 -m pip install --upgrade kaggle
        export PATH="${HOME}/.local/bin:${PATH}"
    fi

    command_exists kaggle || die "Não foi possível encontrar o comando kaggle."
}

check_kaggle_credentials() {
    # Aceita tanto ~/.kaggle/kaggle.json quanto variáveis KAGGLE_USERNAME/KAGGLE_KEY.
    if [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
        chmod 600 "${HOME}/.kaggle/kaggle.json"
        return
    fi

    if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
        return
    fi

    die "Credenciais do Kaggle ausentes. Coloque kaggle.json em ~/.kaggle/kaggle.json ou defina KAGGLE_USERNAME e KAGGLE_KEY."
}

build_baseai() {
    [[ -d "${BASEAI_DIR}" ]] || die "Projeto BaseAI não encontrado em ${BASEAI_DIR}"

    log "Configurando build Release do BaseAI"
    cmake -S "${BASEAI_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

    case "${MODE}" in
        coco)
            log "Compilando gerador de dataset COCO"
            cmake --build "${BUILD_DIR}" \
                --target prepare_detection_dataset \
                -j "${JOBS}"
            ;;

        imagenet)
            log "Compilando gerador de dataset ImageNet"
            cmake --build "${BUILD_DIR}" \
                --target prepare_classification_dataset \
                -j "${JOBS}"
            ;;

        all)
            log "Compilando geradores COCO + ImageNet"
            cmake --build "${BUILD_DIR}" \
                --target prepare_detection_dataset prepare_classification_dataset \
                -j "${JOBS}"
            ;;
    esac
}

find_executable() {
    local name="$1"
    local result=""

    result="$(find "${BUILD_DIR}" -type f -name "${name}" -perm -u+x -print -quit 2>/dev/null || true)"

    if [[ -z "${result}" ]]; then
        result="$(find "${BASEAI_DIR}" -type f -name "${name}" -perm -u+x -print -quit 2>/dev/null || true)"
    fi

    [[ -n "${result}" ]] || die "Executável ${name} não foi encontrado após a compilação."
    printf '%s\n' "${result}"
}

download_imagenet() {
    if [[ -d "${IMAGENET_DIR}/train" && -d "${IMAGENET_DIR}/val" ]] &&
       find "${IMAGENET_DIR}/train" -type f -print -quit | grep -q . &&
       find "${IMAGENET_DIR}/val" -type f -print -quit | grep -q .; then
        log "ImageNet já está organizado em ${IMAGENET_DIR}; download ignorado"
        return
    fi

    mkdir -p "${DOWNLOAD_DIR}/imagenet" "${IMAGENET_DIR}"

    log "Baixando ImageNet pelo Kaggle"
    log "A competição precisa estar aceita na sua conta Kaggle"
    kaggle competitions download \
        -c "${IMAGENET_COMPETITION}" \
        -p "${DOWNLOAD_DIR}/imagenet"

    local competition_zip="${DOWNLOAD_DIR}/imagenet/${IMAGENET_COMPETITION}.zip"
    [[ -f "${competition_zip}" ]] ||
        competition_zip="$(find "${DOWNLOAD_DIR}/imagenet" -maxdepth 1 -type f -name '*.zip' -print -quit)"

    [[ -n "${competition_zip:-}" && -f "${competition_zip}" ]] ||
        die "ZIP do ImageNet não foi encontrado."

    log "Extraindo pacote principal do ImageNet"
    unzip -q -n "${competition_zip}" -d "${DOWNLOAD_DIR}/imagenet/extracted"

    local data_archive=""
    data_archive="$(find "${DOWNLOAD_DIR}/imagenet/extracted" -type f \
        \( -name 'ILSVRC2012_img_train.tar' -o -name 'imagenet_object_localization_patched2019.tar.gz' \) \
        -print -quit || true)"

    if [[ -n "${data_archive}" && "${data_archive}" == *.tar.gz ]]; then
        log "Extraindo arquivo agregado do ImageNet"
        mkdir -p "${DOWNLOAD_DIR}/imagenet/expanded"
        tar -xzf "${data_archive}" -C "${DOWNLOAD_DIR}/imagenet/expanded"
    fi

    # Procura a árvore Data/CLS-LOC entregue pelo Kaggle.
    local cls_root=""
    cls_root="$(find "${DOWNLOAD_DIR}/imagenet" -type d -path '*/Data/CLS-LOC' -print -quit 2>/dev/null || true)"

    if [[ -n "${cls_root}" ]]; then
        mkdir -p "${IMAGENET_DIR}"
        if [[ -d "${cls_root}/train" ]]; then
            rm -rf "${IMAGENET_DIR}/train"
            mv "${cls_root}/train" "${IMAGENET_DIR}/train"
        fi
        if [[ -d "${cls_root}/val" ]]; then
            rm -rf "${IMAGENET_DIR}/val_raw"
            mv "${cls_root}/val" "${IMAGENET_DIR}/val_raw"
        fi
    else
        # Formato alternativo: tar de treino com um tar por classe.
        local train_tar=""
        local val_tar=""
        train_tar="$(find "${DOWNLOAD_DIR}/imagenet" -type f -name 'ILSVRC2012_img_train.tar' -print -quit 2>/dev/null || true)"
        val_tar="$(find "${DOWNLOAD_DIR}/imagenet" -type f -name 'ILSVRC2012_img_val.tar' -print -quit 2>/dev/null || true)"

        [[ -n "${train_tar}" ]] || die "Dados de treino do ImageNet não foram localizados."
        [[ -n "${val_tar}" ]] || die "Dados de validação do ImageNet não foram localizados."

        mkdir -p "${IMAGENET_DIR}/train" "${IMAGENET_DIR}/val_raw"

        log "Extraindo tar principal de treino"
        tar -xf "${train_tar}" -C "${IMAGENET_DIR}/train"

        log "Extraindo os arquivos tar de cada classe"
        find "${IMAGENET_DIR}/train" -maxdepth 1 -type f -name '*.tar' -print0 |
        while IFS= read -r -d '' class_tar; do
            class_name="$(basename "${class_tar}" .tar)"
            mkdir -p "${IMAGENET_DIR}/train/${class_name}"
            tar -xf "${class_tar}" -C "${IMAGENET_DIR}/train/${class_name}"
            rm -f "${class_tar}"
        done

        log "Extraindo validação"
        tar -xf "${val_tar}" -C "${IMAGENET_DIR}/val_raw"
    fi

    [[ -d "${IMAGENET_DIR}/train" ]] || die "Pasta de treino do ImageNet não foi criada."
    [[ -d "${IMAGENET_DIR}/val_raw" ]] || die "Pasta de validação bruta do ImageNet não foi criada."

    local val_csv=""
    val_csv="$(find "${DOWNLOAD_DIR}/imagenet" -type f -name 'LOC_val_solution.csv' -print -quit 2>/dev/null || true)"
    [[ -n "${val_csv}" ]] || die "LOC_val_solution.csv não foi encontrado para organizar a validação."

    log "Organizando validação do ImageNet por classe"
    python3 - "${IMAGENET_DIR}/val_raw" "${IMAGENET_DIR}/val" "${val_csv}" <<'PY'
import csv
import shutil
import sys
from pathlib import Path

raw_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
csv_path = Path(sys.argv[3])

out_dir.mkdir(parents=True, exist_ok=True)

moved = 0
missing = 0

with csv_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row["ImageId"]
        prediction = row["PredictionString"].strip().split()
        if not prediction:
            continue

        synset = prediction[0]
        src = None
        for ext in (".JPEG", ".jpg", ".jpeg", ".png"):
            candidate = raw_dir / f"{image_id}{ext}"
            if candidate.exists():
                src = candidate
                break

        if src is None:
            missing += 1
            continue

        dst_dir = out_dir / synset
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_dir / src.name))
        moved += 1

print(f"Validação organizada: {moved} imagens; ausentes: {missing}")
if moved == 0:
    raise SystemExit("Nenhuma imagem de validação foi organizada.")
PY

    rm -rf "${IMAGENET_DIR}/val_raw"

    if [[ "${KEEP_DOWNLOADS}" != "1" ]]; then
        log "Removendo arquivos compactados temporários do ImageNet"
        rm -rf "${DOWNLOAD_DIR}/imagenet"
    fi
}

download_coco() {
    if [[ -d "${COCO_DIR}/train2017" &&
          -d "${COCO_DIR}/val2017" &&
          -d "${COCO_DIR}/annotations" ]] &&
       find "${COCO_DIR}/train2017" -type f -print -quit | grep -q .; then
        log "COCO já está disponível em ${COCO_DIR}; download ignorado"
        return
    fi

    mkdir -p "${DOWNLOAD_DIR}/coco" "${COCO_DIR}"

    log "Baixando COCO 2017 pelo Kaggle"
    kaggle datasets download \
        -d "${COCO_DATASET}" \
        -p "${DOWNLOAD_DIR}/coco" \
        --unzip

    local coco_root=""
    coco_root="$(find "${DOWNLOAD_DIR}/coco" -type d -name 'coco2017' -print -quit 2>/dev/null || true)"
    [[ -n "${coco_root}" ]] || coco_root="${DOWNLOAD_DIR}/coco"

    local train_dir=""
    local val_dir=""
    local annotations_dir=""

    train_dir="$(find "${coco_root}" -type d -name 'train2017' -print -quit 2>/dev/null || true)"
    val_dir="$(find "${coco_root}" -type d -name 'val2017' -print -quit 2>/dev/null || true)"
    annotations_dir="$(find "${coco_root}" -type d -name 'annotations' -print -quit 2>/dev/null || true)"

    [[ -n "${train_dir}" ]] || die "train2017 não foi encontrado no download do COCO."
    [[ -n "${val_dir}" ]] || die "val2017 não foi encontrado no download do COCO."
    [[ -n "${annotations_dir}" ]] || die "annotations não foi encontrado no download do COCO."

    rm -rf "${COCO_DIR}/train2017" "${COCO_DIR}/val2017" "${COCO_DIR}/annotations"
    mv "${train_dir}" "${COCO_DIR}/train2017"
    mv "${val_dir}" "${COCO_DIR}/val2017"
    mv "${annotations_dir}" "${COCO_DIR}/annotations"

    if [[ "${KEEP_DOWNLOADS}" != "1" ]]; then
        log "Removendo arquivos temporários do COCO"
        rm -rf "${DOWNLOAD_DIR}/coco"
    fi
}

generate_imagenet_shards() {
    if [[ -d "${IMAGENET_SHARDS}" ]] &&
       find "${IMAGENET_SHARDS}" -type f -print -quit | grep -q . &&
       [[ "${FORCE_SHARDS}" != "1" ]]; then
        log "Shards do ImageNet já existem; geração ignorada"
        return
    fi

    [[ -d "${IMAGENET_DIR}/train" ]] || die "Treino do ImageNet ausente."
    [[ -d "${IMAGENET_DIR}/val" ]] || die "Validação do ImageNet ausente."

    local generator
    generator="$(find_executable prepare_classification_dataset)"

    rm -rf "${IMAGENET_SHARDS}"
    mkdir -p "${IMAGENET_SHARDS}"

    log "Gerando shards do ImageNet"
    "${generator}" \
        --train "${IMAGENET_DIR}/train" \
        --val "${IMAGENET_DIR}/val" \
        --output "${IMAGENET_SHARDS}" \
        --train-shards 256 \
        --val-shards 16
}

generate_coco_shards() {
    if [[ -d "${COCO_SHARDS}" ]] &&
       find "${COCO_SHARDS}" -type f -print -quit | grep -q . &&
       [[ "${FORCE_SHARDS}" != "1" ]]; then
        log "Shards do COCO já existem; geração ignorada"
        return
    fi

    [[ -d "${COCO_DIR}/train2017" ]] || die "Treino do COCO ausente."
    [[ -d "${COCO_DIR}/val2017" ]] || die "Validação do COCO ausente."
    [[ -d "${COCO_DIR}/annotations" ]] || die "Anotações do COCO ausentes."

    local generator
    generator="$(find_executable prepare_detection_dataset)"

    rm -rf "${COCO_SHARDS}"
    mkdir -p "${COCO_SHARDS}"

    log "Gerando shards do COCO"
    "${generator}" \
        --train-images "${COCO_DIR}/train2017" \
        --val-images "${COCO_DIR}/val2017" \
        --annotations "${COCO_DIR}/annotations" \
        --output "${COCO_SHARDS}" \
        --train-shards 128 \
        --val-shards 8
}

show_summary() {
    log "Resumo"
    [[ -d "${IMAGENET_DIR}" ]] && du -sh "${IMAGENET_DIR}" || true
    [[ -d "${IMAGENET_SHARDS}" ]] && du -sh "${IMAGENET_SHARDS}" || true
    [[ -d "${COCO_DIR}" ]] && du -sh "${COCO_DIR}" || true
    [[ -d "${COCO_SHARDS}" ]] && du -sh "${COCO_SHARDS}" || true
    df -h "${WORKSPACE}" || true
}

case "${MODE}" in
    imagenet|coco|all)
        ;;
    *)
        die "Modo inválido: ${MODE}. Use imagenet, coco ou all."
        ;;
esac

mkdir -p "${WORKSPACE}" "${DOWNLOAD_DIR}"
install_dependencies
check_kaggle_credentials
build_baseai

case "${MODE}" in
    imagenet)
        download_imagenet
        generate_imagenet_shards
        ;;
    coco)
        download_coco
        generate_coco_shards
        ;;
    all)
        download_imagenet
        generate_imagenet_shards
        download_coco
        generate_coco_shards
        ;;
esac

show_summary
log "Processo concluído com sucesso"
