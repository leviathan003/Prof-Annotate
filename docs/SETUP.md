# Setup Instructions
To setup and start the application the user may follow multiple approaches:

1. Using the venv and source code (Manual Approach): If the user has cloned the repo, it is necessary to setup the venv with all the necessary dependencies, after which the user can run main.py to launch the application.
Following are the steps for the same:

1.1. Open a terminal in the cloned repository directory. 
</br>
1.2. Make a venv using python and activate the venv
```
bash

python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```
1.3. Upgrade pip
```
bash

pip install --upgrade pip
```
1.4. Install dependencies

CPU-only (default):
```
bash

pip install -r requirements.txt
```
GPU (CUDA) — replaces the CPU onnxruntime:

```
bash

pip install -r requirements.txt
pip uninstall -y onnxruntime
pip install onnxruntime-gpu>=1.17.0
```

1.5. Place the ONNX model

Download or copy yolo11n_segpose.onnx into the models/ directory:
```
bash

profannotate/
└── models/
    └── yolo11n_segpose.onnx # Currently included in the repository
```
The application will start without it, but auto-annotation will not work.
1.6. Run
```
bash

python main.py
```
2. Using the setup script (Automated Setup): The user can also directly setup the env by running the setup script in the /scripts/setup.sh dir.
```
bash

bash scripts/setup.sh                # auto: gpu-cuda12 if an NVIDIA GPU is found, else cpu
bash scripts/setup.sh --cpu          # force CPU-only onnxruntime (universal)
bash scripts/setup.sh --gpu          # modern NVIDIA: onnxruntime-gpu, CUDA 12 / cuDNN 9
bash scripts/setup.sh --gpu-cuda12   # same as --gpu
bash scripts/setup.sh --gpu-cuda11   # legacy NVIDIA: onnxruntime-gpu 1.18, CUDA 11.8 / cuDNN 8
bash scripts/setup.sh --dev          # + dev tools (pytest, black, ruff)
bash scripts/setup.sh --gpu --dev
```

3. Using the installed AppImage binary: If the user has downloaded an AppImage binary from the Releases section, then they may double click on the binary file to start it. If the binary file is not starting even after double clicking it, please open a terminal window in the same folder and type in the following:
```
bash

chmod +x ProfAnnotate-cpu-x86_64.AppImage #For cpu build
chmod +x ProfAnnotate-gpu-cuda12-x86_64.AppImage #For cuda12 build
chmod +x ProfAnnotate-gpu-cuda11-x86_64.AppImage #For cuda11 build
```
to allow executable permissions to the binary. The user can now double click on the binary to start it.

The releases are of three variants, CPU, GPU(CUDA 12) and GPU(CUDA 11 - for older devices). The CPU build is the universal build, it is made to run on any machine with a Linux distro that is x86_64. However the functioning is only been tested on Ubuntu, Debian and Arch variants with testing on other variants like Fedora, RHEL, to be tested. The user is welcome to try it out and raise any issues regarding the same. 

However there must not be a misconception that the GPU builds will not work on CPU only machines, they are build with in-built fallback mechanisms to CPU mode if a GPU is not detected.

After complete setup the user will be greeted by the following splashscreen on starting the application.

![App Screenshot](docs/images/splashscreen.png)
