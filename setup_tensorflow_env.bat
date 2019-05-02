@echo OFF

if %1.==. goto No1

set CurrentPath=%cd%
set ModulePath=%1
set EnvironmentPath=%ModulePath%\Env

:: Create the environment and activate it

call conda create -y --name %EnvironmentPath% tensorflow-gpu
call activate %EnvironmentPath%

call conda install -y opencv=3.3
call pip install tensorflow_hub

call git clone -b pyIGTLink_client https://github.com/Sunderlandkyl/pyIGTLink.git %ModulePath%\pyIGTLink
call pip install -e %ModulePath%\pyIGTLink

GOTO End1

:No1
  echo.
  echo Usage: %~n0 PROJECT_PATH
  echo E.g.: %~n0 c:\MyProject
  echo.
  echo Note: If admin access is needed to write the environment path, then make sure to start this Anaconda Prompt in Administrator mode.
  echo.
goto End1

:End1