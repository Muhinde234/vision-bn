# Zips the YOLO dataset for upload to Google Drive
# Run from the vision-backend directory:
#   .\prepare_colab_upload.ps1

$root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$src     = Join-Path $root "malaria\yolo_dataset"
$zipOut  = Join-Path $root "malaria_yolo_dataset.zip"

if (Test-Path $zipOut) { Remove-Item $zipOut -Force }

Write-Host "Zipping $src ..."
Write-Host "(This may take a few minutes for 2 GB of images)"

Compress-Archive -Path "$src\*" -DestinationPath $zipOut -CompressionLevel Fastest

$sizeMB = [math]::Round((Get-Item $zipOut).Length / 1MB, 1)
Write-Host ""
Write-Host "Done: $zipOut  ($sizeMB MB)"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Upload malaria_yolo_dataset.zip to your Google Drive"
Write-Host "  2. Open malaria_colab_training.ipynb in Google Colab"
Write-Host "  3. Run all cells (Runtime > Run all)"
Write-Host "  4. Download best.pt from Drive when training finishes"
