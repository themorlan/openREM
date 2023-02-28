@echo off
if not exist "consumer-files" mkdir consumer-files >nul
echo ##############################
echo # OpenREM Task Queue Starter #
echo ##############################
echo.
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~
echo ~ Removing old consumers ~
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~
echo.
set test=
for /f "tokens=1*" %%a in ('
    sc query state^= all ^| findstr /r /c:"SERVICE_NAME: huey-consumer-.*"
') do (
    echo - Stopping %%b
    net stop %%b >nul 2>&1
    echo - Deleting %%b service
    sc delete %%b >nul
    echo - Deleting %%b files
    del consumer-files\%%b* >nul
    set test=y
)
if [%test%]==[] echo Nothing to do!
echo.
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo ~ Configuring new consumers ~
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo.
set "domain=."
set /p "domain=Enter domain or just ENTER for default [.] : "
set /p "username=Enter username: "
set /p "password=Enter password: "
set "workers=1"
set /p "workers=How many workers would you like to spawn? [1] : "
echo.
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo ~ Instantiating new consumers ~
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo.
for /l %%x in (1, 1, %workers%) do (
    (
        echo ^<service^>
        echo   ^<id^>huey-consumer-%%x^</id^>
        echo   ^<name^>HUEY CONSUMER %%x^</name^>
        echo   ^<description^>This service runs a huey consumer.^</description^>
        echo   ^<executable^>E:\venv\Scripts\python.exe^</executable^>
        echo   ^<arguments^>E:\openrem\openrem\manage.py run_huey^</arguments^>
        echo   ^<onfailure action="restart" /^>
        echo   ^<startmode^>Automatic^</startmode^>
        echo   ^<serviceaccount^>
	    echo     ^<domain^>%domain%^</domain^>
        echo     ^<user^>%username%^</user^>
        echo     ^<password^>%password%^</password^>
        echo     ^<allowservicelogon^>true^</allowservicelogon^>
        echo   ^</serviceaccount^>
        echo ^</service^>
    ) > "consumer-files\huey-consumer-%%x.xml"
    echo Preparing huey-consumer-%%x...
    COPY WinSW.exe consumer-files\huey-consumer-%%x.exe >nul
    start /B /W "" consumer-files\huey-consumer-%%x.exe stopwait >nul
    start /B /W "" consumer-files\huey-consumer-%%x.exe uninstall >nul
    echo Installing huey-consumer-%%x as service...
    start /B /W "" consumer-files\huey-consumer-%%x.exe install >nul
    echo Starting huey-consumer-%%x service...
    start /B /W "" consumer-files\huey-consumer-%%x.exe start >nul
)
echo.
echo All done, you can now safely close this window!
echo.
pause