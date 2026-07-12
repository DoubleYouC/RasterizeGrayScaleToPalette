# build.ps1
$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
& "env\Scripts\Activate.ps1"

Write-Host "Running PyInstaller with spec file..."
pyinstaller "RasterizeGrayScaleToPalette.spec"

Write-Host "Build complete. Output is in the dist/ folder."
