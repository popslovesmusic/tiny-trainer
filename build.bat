@echo off
set compiler_flags=/EHsc /nologo /W4
set include_path=/I"core"
set source_files=main.cpp core\Translator.cpp
set output_file=tiny-agent-trainer.exe

echo Compiling C++ files...
cl.exe %compiler_flags% %include_path% %source_files% /Fe:%output_file%

if %errorlevel% neq 0 (
    echo.
    echo BUILD FAILED
    exit /b %errorlevel%
)

echo.
echo BUILD SUCCESSFUL
echo Executable created: %output_file%