from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("PySide6")
binaries = collect_dynamic_libs("PySide6")

# Qt plugins needed for GUI + fonts
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("PySide6")
