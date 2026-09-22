[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Runtime,
    [string]$Output = "dist/runtime",
    [string]$CudaRoot = "",
    [string]$VsRedistRoot = "",
    [string]$CudaArchitectures = "",
    [switch]$AllowUncommitted
)

$ErrorActionPreference = "Stop"

function Find-CudaRoot {
    param([string]$Requested)
    foreach ($candidate in @($Requested, $env:CUDA_PATH, $env:CUDAToolkit_ROOT)) {
        if ($candidate -and (Test-Path -LiteralPath (Join-Path $candidate "bin/nvcc.exe"))) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $installRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $installRoot) {
        $found = Get-ChildItem -LiteralPath $installRoot -Directory |
            Sort-Object Name -Descending |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "bin/nvcc.exe") } |
            Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    throw "CUDA Toolkit was not found; pass -CudaRoot with the toolkit used to build the worker"
}

function Find-VsRedistRoot {
    param([string]$Requested)
    if ($Requested) {
        $path = (Resolve-Path -LiteralPath $Requested).Path
        if (-not (Test-Path -LiteralPath (Join-Path $path "x64/Microsoft.VC143.CRT"))) {
            throw "-VsRedistRoot must point to a VC/Redist/MSVC/<version> directory"
        }
        return $path
    }
    foreach ($programRoot in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $programRoot) { continue }
        $vsRoot = Join-Path $programRoot "Microsoft Visual Studio/2022"
        if (-not (Test-Path -LiteralPath $vsRoot)) { continue }
        foreach ($edition in (Get-ChildItem -LiteralPath $vsRoot -Directory)) {
            $redist = Join-Path $edition.FullName "VC/Redist/MSVC"
            if (-not (Test-Path -LiteralPath $redist)) { continue }
            $found = Get-ChildItem -LiteralPath $redist -Directory |
                Where-Object { $_.Name -match '^\d+\.\d+' } |
                Sort-Object Name -Descending |
                Where-Object {
                    (Test-Path -LiteralPath (Join-Path $_.FullName "x64/Microsoft.VC143.CRT")) -and
                    (Test-Path -LiteralPath (Join-Path $_.FullName "x64/Microsoft.VC143.OpenMP"))
                } | Select-Object -First 1
            if ($found) { return $found.FullName }
        }
    }
    throw "Visual Studio 2022 x64 VC and OpenMP redistributables were not found; pass -VsRedistRoot"
}

function Copy-RequiredPattern {
    param([string]$Directory, [string]$Pattern, [string]$Destination)
    $foundFiles = @(Get-ChildItem -LiteralPath $Directory -File -Filter $Pattern -ErrorAction SilentlyContinue)
    if ($foundFiles.Count -ne 1) {
        throw "Expected exactly one $Pattern in $Directory; found $($foundFiles.Count)"
    }
    Copy-Item -LiteralPath $foundFiles[0].FullName -Destination $Destination -Force
}

$repo = Split-Path $PSScriptRoot -Parent
$package = Join-Path $repo "packaging/runtime"
$native = Join-Path $package "src/fishs2rt_runtime/_native"
$worker = Join-Path $native "audiocpp_cli.exe"
$manifest = Join-Path $package "src/fishs2rt_runtime/runtime-manifest.json"
$buildInfoFile = Join-Path $package "src/fishs2rt_runtime/BUILD_INFO.txt"
$licenses = Join-Path $package "src/fishs2rt_runtime/licenses"
$resolvedRuntime = (Resolve-Path -LiteralPath $Runtime).Path
$cache = Join-Path (Split-Path (Split-Path $resolvedRuntime -Parent) -Parent) "CMakeCache.txt"
if (-not $CudaRoot -and (Test-Path -LiteralPath $cache)) {
    $toolkitEntry = Get-Content -LiteralPath $cache |
        Where-Object { $_ -match '^CUDAToolkit_ROOT:[^=]*=' } |
        Select-Object -First 1
    if ($toolkitEntry) { $CudaRoot = ($toolkitEntry -split '=', 2)[1] }
}
$resolvedOutput = if ([System.IO.Path]::IsPathRooted($Output)) {
    [System.IO.Path]::GetFullPath($Output)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repo $Output))
}
$resolvedCudaRoot = Find-CudaRoot $CudaRoot
$resolvedVsRedistRoot = Find-VsRedistRoot $VsRedistRoot

New-Item -ItemType Directory -Path $native -Force | Out-Null
New-Item -ItemType Directory -Path $licenses -Force | Out-Null
New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null

try {
    Copy-Item -LiteralPath $resolvedRuntime -Destination $worker -Force
    $buildInfoJson = & $worker --build-info-json
    if ($LASTEXITCODE -ne 0) { throw "runtime --build-info-json failed" }
    $buildInfo = ($buildInfoJson -join "`n") | ConvertFrom-Json
    if ($buildInfo.product -ne "FishS2RT") { throw "runtime product is not FishS2RT" }
    if ($buildInfo.runtime_version -ne "0.1.0") { throw "runtime version must be 0.1.0" }
    if ($buildInfo.protocol_version -ne 1) { throw "runtime protocol must be 1" }
    if (-not $AllowUncommitted -and [string]::IsNullOrWhiteSpace($buildInfo.source_commit)) {
        throw "runtime source commit is missing; commit source and rebuild before publishing"
    }
    if (-not $AllowUncommitted) {
        $repoCommit = (& git -C $repo rev-parse --short HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $buildInfo.source_commit -ne $repoCommit) {
            throw "runtime source commit does not match the current Git HEAD; rebuild the worker"
        }
        $dirty = & git -C $repo status --porcelain
        if ($LASTEXITCODE -ne 0 -or $dirty) {
            throw "source tree must be committed and clean before building a release wheel"
        }
    }
    $cacheArchitectures = ""
    if (Test-Path -LiteralPath $cache) {
        $entry = Get-Content -LiteralPath $cache |
            Where-Object { $_ -match '^CMAKE_CUDA_ARCHITECTURES:[^=]*=' } |
            Select-Object -First 1
        if ($entry) { $cacheArchitectures = ($entry -split '=', 2)[1] }
    }
    if ($CudaArchitectures -and $cacheArchitectures -and $CudaArchitectures -ne $cacheArchitectures) {
        throw "-CudaArchitectures disagrees with the worker's CMakeCache.txt"
    }
    if (-not $CudaArchitectures) { $CudaArchitectures = $cacheArchitectures }
    if (-not $CudaArchitectures) {
        throw "CUDA architectures are unknown; pass -CudaArchitectures explicitly"
    }
    $buildInfo | Add-Member -NotePropertyName cuda_architectures -NotePropertyValue $CudaArchitectures
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($manifest, (($buildInfo | ConvertTo-Json -Compress) + "`n"), $utf8)
    $buildDescription = @(
        "FishS2RT runtime $($buildInfo.runtime_version)",
        "Git source commit: $($buildInfo.source_commit)",
        "audio.cpp upstream commit: $($buildInfo.upstream_commit)",
        "Protocol version: $($buildInfo.protocol_version)",
        "CUDA architectures: $CudaArchitectures",
        "CUDA Toolkit version: $(Split-Path $resolvedCudaRoot -Leaf)",
        "MSVC redistributables version: $(Split-Path $resolvedVsRedistRoot -Leaf)"
    ) -join "`n"
    [System.IO.File]::WriteAllText($buildInfoFile, ($buildDescription + "`n"), $utf8)
    foreach ($capability in @("streaming", "reference_cache", "q8", "bf16")) {
        if ($buildInfo.capabilities -notcontains $capability) {
            throw "runtime capability is missing: $capability"
        }
    }
    Copy-Item -LiteralPath (Join-Path $repo "LICENSE") -Destination (Join-Path $licenses "Apache-2.0.txt") -Force
    Copy-Item -LiteralPath (Join-Path $repo "NOTICE") -Destination (Join-Path $licenses "NOTICE.txt") -Force
    Copy-Item -LiteralPath (Join-Path $repo "THIRD_PARTY_NOTICES.md") -Destination $licenses -Force
    Copy-Item -LiteralPath (Join-Path $repo "licenses/FISH_AUDIO_RESEARCH_LICENSE.md") -Destination $licenses -Force
    Copy-Item -LiteralPath (Join-Path $repo "licenses/third-party") -Destination $licenses -Recurse -Force
    $runtimeLicenses = Join-Path $licenses "runtime"
    New-Item -ItemType Directory -Path $runtimeLicenses -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $resolvedCudaRoot "EULA.txt") -Destination (Join-Path $runtimeLicenses "NVIDIA_CUDA_EULA.txt") -Force
    Copy-Item -LiteralPath (Join-Path $repo "licenses/runtime/MICROSOFT_VISUAL_STUDIO_REDIST.txt") -Destination $runtimeLicenses -Force

    $cudaBin = Join-Path $resolvedCudaRoot "bin"
    foreach ($pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "cufft64_*.dll")) {
        Copy-RequiredPattern $cudaBin $pattern $native
    }
    $redistX64 = Join-Path $resolvedVsRedistRoot "x64"
    foreach ($name in @("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll")) {
        Copy-Item -LiteralPath (Join-Path $redistX64 "Microsoft.VC143.CRT/$name") -Destination $native -Force
    }
    Copy-Item -LiteralPath (Join-Path $redistX64 "Microsoft.VC143.OpenMP/VCOMP140.DLL") -Destination $native -Force
    $stagedInfo = (& $worker --build-info-json) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $stagedInfo.product -ne $buildInfo.product -or
        $stagedInfo.runtime_version -ne $buildInfo.runtime_version) {
        throw "staged worker did not start with its packaged DLLs"
    }

    Push-Location $env:TEMP
    try {
        python -m build --wheel --outdir $resolvedOutput $package
        if ($LASTEXITCODE -ne 0) { throw "runtime wheel build failed" }
    }
    finally {
        Pop-Location
    }
}
finally {
    Remove-Item -LiteralPath $worker -Force -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $native -File -Filter "*.dll" -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $manifest -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $buildInfoFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $licenses -Recurse -Force -ErrorAction SilentlyContinue
}
