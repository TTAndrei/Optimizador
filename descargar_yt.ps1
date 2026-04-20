param(
    [string]$Url,
    [switch]$SoloAudio
)

$ErrorActionPreference = "Stop"

$outputDir = "C:\Users\andre\Desktop\youtube"

function Test-CommandExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName
    )

    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

if (-not (Test-Path -Path $outputDir)) {
    New-Item -Path $outputDir -ItemType Directory -Force | Out-Null
}

if (-not (Test-CommandExists -CommandName "yt-dlp")) {
    Write-Host "Error: yt-dlp no esta disponible en PATH." -ForegroundColor Red
    Write-Host "Instalalo o agrega su ruta al PATH y vuelve a intentar." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-CommandExists -CommandName "ffmpeg")) {
    Write-Host "Error: ffmpeg no esta disponible en PATH." -ForegroundColor Red
    Write-Host "ffmpeg es necesario para convertir formatos de salida automaticamente." -ForegroundColor Yellow
    exit 1
}

if ([string]::IsNullOrWhiteSpace($Url)) {
    $Url = Read-Host "Pega la URL del video"
}

if ([string]::IsNullOrWhiteSpace($Url)) {
    Write-Host "No se proporciono URL. Saliendo..." -ForegroundColor Red
    exit 1
}

if (-not $PSBoundParameters.ContainsKey("SoloAudio")) {
    $respuesta = (Read-Host "Descargar solo audio? (s/n)").Trim().ToLowerInvariant()
    $SoloAudio = $respuesta -in @("s", "si", "y", "yes")
}

$commonArgs = @(
    "--newline",
    "-P", $outputDir,
    "-o", "%(title)s [%(id)s].%(ext)s",
    "--no-playlist"
)

if ($SoloAudio) {
    Write-Host "Descargando solo audio en $outputDir ..." -ForegroundColor Cyan
    $ytArgs = @(
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--embed-metadata",
        "--add-metadata",
        $Url
    )
}
else {
    Write-Host "Descargando video en maxima calidad y convirtiendo a mp4 en $outputDir ..." -ForegroundColor Cyan
    $ytArgs = @(
        "-f", "bv*+ba/b",
        "--recode-video", "mp4",
        "--merge-output-format", "mp4",
        "--embed-metadata",
        "--add-metadata",
        $Url
    )
}

& yt-dlp @commonArgs @ytArgs
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "Descarga finalizada correctamente." -ForegroundColor Green
    Write-Host "Archivos guardados en: $outputDir"
    exit 0
}

Write-Host "yt-dlp termino con codigo de salida: $exitCode" -ForegroundColor Red
exit $exitCode