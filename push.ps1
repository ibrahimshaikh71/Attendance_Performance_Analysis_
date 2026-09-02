#!/usr/bin/env powershell
# Simple GitHub Push Script
# Run this script to push your Attendance Project to GitHub

$ErrorActionPreference = "Continue"

Write-Host "`n" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "GitHub Push - Attendance Project" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`n"

$projectPath = "c:\Users\Admin\Downloads\Attendance_Performance_Analysis_Project"
Set-Location $projectPath

# Show current status
Write-Host "📁 Project Path: $projectPath" -ForegroundColor Cyan
Write-Host "`n"

# Check if there are commits to push
Write-Host "Checking commits to push..." -ForegroundColor Yellow
Write-Host "`n"

# Configure credential helper
Write-Host "✓ Configuring Git..." -ForegroundColor Green
git config --global credential.helper wincred
git config --global credential.useHttpPath true

Write-Host "`n"
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "NEXT: When you see a browser window" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "`n"
Write-Host "1. Complete GitHub authentication in browser" -ForegroundColor White
Write-Host "2. Return to this terminal" -ForegroundColor White
Write-Host "3. Press Enter to continue" -ForegroundColor White
Write-Host "`n"

Read-Host "Press Enter when browser authentication is complete"

Write-Host "`n"
Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Cyan
Write-Host "`n"

git push origin main --verbose

Write-Host "`n"
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Push Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`n"
Write-Host "View your repository:" -ForegroundColor Green
Write-Host "https://github.com/ibrahimshaikh71/Attendance_Performance_Analysis_Project" -ForegroundColor Cyan
Write-Host "`n"

Read-Host "Press Enter to close"
