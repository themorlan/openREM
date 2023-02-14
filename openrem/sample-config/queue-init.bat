@echo off
set "domain=."
set /p "domain=Enter domain or just ENTER for default [.] : "
set /p "username=Enter username: "
set /p "password=Enter password: "
set "workers=1"
set /p "workers=How many workers would you like to spawn? [1] : "

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
    ) > "huey_worker_%%x.xml"
    COPY WinSW.exe huey_worker_%%x.exe >nul
    start /W "" huey_worker_%%x.exe stopwait
    start /W "" huey_worker_%%x.exe uninstall
::    start /W "" huey_worker_%%x.exe install
::    start /W "" huey_worker_%%x.exe start
)

pause