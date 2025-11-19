#run FastAPI

uvicorn main:app --reload --port 8000

# If the server is not killed, and you try to start it again. port 8000 is already taken, the request will hang. to kill the process, in windows powershell

Stop-Process -Name python -Force

Get-Process python*

Run uvicorn main:app --reload --port 8000 and hit GET /api/generate-hedge-factor-report to verify data and base64 payloads.
