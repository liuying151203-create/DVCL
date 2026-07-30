#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this script with: source scripts/activate_gpu_env.sh" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="${DVCL_GPU_ENV:-${repo_root}/.conda/dvcl-cu121-py39}"

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  echo "DVCL GPU environment not found: ${env_prefix}" >&2
  return 1
fi

conda_executable="${CONDA_EXE:-$(command -v conda 2>/dev/null)}"
if [[ -n "${conda_executable}" ]]; then
  eval "$("${conda_executable}" shell.bash hook)"
  conda activate "${env_prefix}"
else
  export PATH="${env_prefix}/bin:${PATH}"
  export CONDA_PREFIX="${env_prefix}"
fi

python_site="$("${env_prefix}/bin/python" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cuda_library_paths=("${python_site}/torch/lib")
for library_path in "${python_site}"/nvidia/*/lib; do
  if [[ -d "${library_path}" ]]; then
    cuda_library_paths+=("${library_path}")
  fi
done

export DVCL_GPU_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
joined_cuda_paths="$(IFS=:; echo "${cuda_library_paths[*]}")"
export LD_LIBRARY_PATH="${joined_cuda_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"

echo "Activated DVCL GPU environment: ${env_prefix}"
python -c \
  'import torch, dgl, torch_geometric; print(f"Torch {torch.__version__}; DGL {dgl.__version__}; PyG {torch_geometric.__version__}; CUDA available: {torch.cuda.is_available()}")'
