@echo off
REM GitHub Push Script for Windows
REM This script handles GitHub authentication and pushes your project

cd /d "c:\Users\Admin\Downloads\Attendance_Performance_Analysis_Project"

echo.
echo ========================================
echo     Attendance Project - GitHub Push
echo ========================================
echo.

REM Configure git credential helper
git config --global credential.helper wincred
git config --global credential.useHttpPath true

echo Configured Git credential helper.
echo.
echo Please follow the browser authentication window that will appear.
echo.
echo Starting push...
echo.

REM Push to GitHub
git push -u origin main --verbose

echo.
echo ========================================
echo Push complete!
echo ========================================
echo.
echo Your repository:
echo https://github.com/ibrahimshaikh71/Attendance_Performance_Analysis_Project
echo.

pause
