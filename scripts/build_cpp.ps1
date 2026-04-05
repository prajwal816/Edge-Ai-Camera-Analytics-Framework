$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BuildDir = Join-Path $Root "build"
cmake -S $Root -B $BuildDir -DCMAKE_BUILD_TYPE=Release
cmake --build $BuildDir --parallel
$EngineDir = Join-Path $Root "engine"
New-Item -ItemType Directory -Force -Path $EngineDir | Out-Null
Get-ChildItem (Join-Path $BuildDir "python_module") -Filter "edge_infer_native*.pyd" -ErrorAction SilentlyContinue |
    Copy-Item -Destination $EngineDir -Force
Get-ChildItem (Join-Path $BuildDir "python_module") -Filter "edge_infer_native*.so" -ErrorAction SilentlyContinue |
    Copy-Item -Destination $EngineDir -Force
Write-Host "Native module copied to engine/ (if build succeeded)."
