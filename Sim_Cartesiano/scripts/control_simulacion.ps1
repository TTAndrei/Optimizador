# Script de ayuda para controlar la simulación desde PowerShell
# Uso: .\control_simulacion.ps1 [plot|stop]

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("plot", "stop")]
    [string]$Accion
)

switch ($Accion) {
    "plot" {
        Write-Host "📊 Creando trigger para generar gráficos..." -ForegroundColor Cyan
        New-Item -Path "PLOT_NOW.trigger" -ItemType File -Force | Out-Null
        Write-Host "✓ Trigger creado. La simulación generará gráficos en breve." -ForegroundColor Green
        Write-Host "  (Se verificará en la próxima iteración múltiplo de 10)" -ForegroundColor Gray
    }
    "stop" {
        Write-Host "🛑 Creando trigger para detener simulación..." -ForegroundColor Yellow
        New-Item -Path "STOP_SIMULATION.trigger" -ItemType File -Force | Out-Null
        Write-Host "✓ Trigger creado. La simulación se detendrá limpiamente." -ForegroundColor Green
        Write-Host "  (Se detendrá en la próxima iteración múltiplo de 10)" -ForegroundColor Gray
    }
}

Write-Host "`nMonitorea la consola de la simulación para ver cuándo se procesa el trigger." -ForegroundColor Gray
