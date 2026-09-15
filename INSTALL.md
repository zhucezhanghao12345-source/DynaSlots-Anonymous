# Installation

The reference environment uses Ubuntu 20.04, Python 3.8, CUDA-capable NVIDIA
GPUs, and MuJoCo 2.1. Commands below assume the repository is named
`DynaSlots`.

## 1. Create the Python environment

```bash
git clone https://github.com/YOUR_GITHUB_ACCOUNT/DynaSlots.git
cd DynaSlots

conda create -n dynaslots python=3.8 -y
conda activate dynaslots
python -m pip install --upgrade pip
```

Install the PyTorch build matching your CUDA driver. For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the Python dependencies and DynaSlots itself:

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Install MuJoCo 2.1

Install the system build dependencies:

```bash
sudo apt update
sudo apt install -y \
  libx11-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  libxi-dev libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev \
  libglew-dev libvulkan1 patchelf
```

Download MuJoCo 2.1:

```bash
mkdir -p "$HOME/.mujoco"
cd "$HOME/.mujoco"
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz
tar -xzf mujoco210.tar.gz
cd -
```

Add these variables to your shell configuration, then open a new shell:

```bash
export MUJOCO_HOME="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_HOME/bin:/usr/lib/nvidia:/usr/local/cuda/lib64"
export MUJOCO_GL=egl
```

Install the bundled MuJoCo Python binding:

```bash
pip install -e third_party/mujoco-py-2.1.2.14
```

## 3. Install the simulation dependencies

The repository carries the exact dependency sources used by the release. It
does not track build directories, binary extensions, datasets, or pretrained
weights.

```bash
pip install setuptools==80.10.2 Cython==0.29.35 patchelf==0.17.2.0
pip install -e third_party/gym-0.21.0
pip install -e third_party/Metaworld
pip install -e third_party/rrl-dependencies/mj_envs
pip install -e third_party/rrl-dependencies/mjrl
pip install --no-build-isolation -e third_party/pytorch3d_simplified
```

## 4. Adroit expert checkpoints

Download the VRL3 Adroit experts from
[OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS) and
place them as follows:

```text
third_party/VRL3/ckpts/
├── vrl3_door.pt
├── vrl3_hammer.pt
├── vrl3_pen.pt
└── vrl3_relocate.pt
```

The checkpoint directory is ignored by Git.

## 5. Verify the installation

```bash
python -c "import dynaslots; print(dynaslots.__version__)"
pytest -q tests/test_dynaslots.py
```

If MuJoCo cannot create an OpenGL context on a headless machine, verify that
`MUJOCO_GL=egl` is exported and that the NVIDIA EGL libraries are visible in
`LD_LIBRARY_PATH`.
