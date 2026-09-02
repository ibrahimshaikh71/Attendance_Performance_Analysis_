@echo off
REM GitHub Push Helper Script for Attendance Project
REM This script automates the git push process

cd /d "c:\Users\Admin\Downloads\Attendance_Performance_Analysis_Project"

echo.
echo ========================================
echo GitHub Push Setup for Attendance Project
echo ========================================
echo.

REM Display current status
echo Current Git Status:
git status
echo.

REM Configure git to store credentials
echo Configuring Git credential helper...
git config --global credential.helper wincred
git config --global credential.useHttpPath true

echo.
echo ========================================
echo NEXT STEP - Choose Authentication Method
echo ========================================
echo.
echo Option 1: Use Personal Access Token
echo   - Go to: https://github.com/settings/tokens
echo   - Click: Generate new token (classic)
echo   - Scopes: repo, workflow
echo   - Copy the token and paste when prompted
echo.
echo Option 2: Use SSH (if configured)
echo   - SSH keys must be added to GitHub
echo.
echo Running push now...
echo When asked for username: ibrahimshaikh71
echo When asked for password: Paste your token or use SSH key
echo.
pause

git push origin main -v

echo.
echo Push complete! Check your GitHub repository:
echo https://github.com/ibrahimshaikh71/Attendance_Performance_Analysis_Project
echo.
pause
