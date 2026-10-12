$VenvName = "ai-cam"

# Create the virtual environment if it doesn't exist
if (-Not (Test-Path "$VenvName\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment '$VenvName'..."
    python -m venv $VenvName
}

# Dot-source the activation script so it applies to the current terminal session
Write-Host "Activating '$VenvName'..."
. .\$VenvName\Scripts\Activate.ps1