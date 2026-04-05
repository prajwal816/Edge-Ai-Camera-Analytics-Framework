$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BuildDir = Join-Path $Root "build"
cmake -S $Root -B $BuildDir -DCMAKE_BUILD_TYPE=Release
cmake --build $BuildDir --parallel
$EngineDir = Join-Path $Root "engine"
New-Item -ItemType Directory -Force -Path $EngineDir | Out-Null
foreach ($sub in @("python_module", "python_module\Debug", "python_module\Release")) {
    $dir = Join-Path $BuildDir $sub
    Get-ChildItem $dir -Filter "edge_infer_native*.pyd" -ErrorAction SilentlyContinue |
        Copy-Item -Destination $EngineDir -Force
    Get-ChildItem $dir -Filter "edge_infer_native*.so" -ErrorAction SilentlyContinue |
        Copy-Item -Destination $EngineDir -Force
}
Write-Host "Native module copied to engine/ (if build succeeded)."
