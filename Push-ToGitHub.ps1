# PowerShell script to push Attendance Project to GitHub
# Run this after generating your Personal Access Token

param(
    [Parameter(Mandatory=$false)]
    [string]$Token = "",
    [Parameter(Mandatory=$false)]
    [string]$Username = "ibrahimshaikh71"
)

$projectPath = "c:\Users\Admin\Downloads\Attendance_Performance_Analysis_Project"
cd $projectPath

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Push - Attendance Project" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check current status
Write-Host "Current Status:" -ForegroundColor Yellow
git status --short
Write-Host ""

# Configure credential helper
Write-Host "Configuring Git credential helper..." -ForegroundColor Yellow
git config --global credential.helper wincred
git config --global credential.useHttpPath true

if ($Token) {
    # Store credentials
    Write-Host "Storing credentials..." -ForegroundColor Yellow
    $url = "https://$Username`:$Token@github.com"
    git credential approve <<< @"
protocol=https
host=github.com
username=$Username
password=$Token
"@
}

# Attempt push
Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
Write-Host ""

git push origin main -v

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Push Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your repository: https://github.com/$Username/Attendance_Performance_Analysis_Project" -ForegroundColor Green
Write-Host ""
Write-Host "To verify:" -ForegroundColor Cyan
Write-Host "  git log --oneline" -ForegroundColor Gray
Write-Host "  git remote -v" -ForegroundColor Gray
