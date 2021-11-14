chcp 65001

@echo off

REM invoke anaconda prompt
set root=C:\Users\Kry\Anaconda3
call %root%\Scripts\activate.bat %root%

REM run tensorboard logging
call tensorboard --logdir="C:\Krisz\BME\MSc\Semester 1\Önlab 1\buggy\logs\a2c"