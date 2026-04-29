@echo off
chcp 65001 > nul
echo ====================================
echo  강의 녹음 자동화 - 스케줄러 등록
echo ====================================

:: 현재 스크립트 위치
set SCRIPT_DIR=%~dp0
set PYTHON_SCRIPT=%SCRIPT_DIR%main.py

:: Python 경로 확인
for /f "tokens=*" %%i in ('where python 2^>nul') do set PYTHON_PATH=%%i

if "%PYTHON_PATH%"=="" (
    echo [오류] Python이 설치되어 있지 않거나 PATH에 없습니다.
    pause
    exit /b 1
)

echo Python 경로: %PYTHON_PATH%
echo 스크립트 경로: %PYTHON_SCRIPT%
echo.

:: 작업 스케줄러 등록 (매 1시간마다 실행)
schtasks /create /tn "강의녹음자동화" ^
    /tr "\"%PYTHON_PATH%\" \"%PYTHON_SCRIPT%\"" ^
    /sc hourly ^
    /mo 1 ^
    /st 09:00 ^
    /ru "%USERNAME%" ^
    /f

if %errorlevel% == 0 (
    echo.
    echo [성공] 매 1시간마다 자동 실행되도록 등록되었습니다.
    echo 작업 스케줄러에서 '강의녹음자동화' 항목으로 확인 가능합니다.
) else (
    echo [실패] 스케줄러 등록에 실패했습니다. 관리자 권한으로 실행해주세요.
)

echo.
pause
