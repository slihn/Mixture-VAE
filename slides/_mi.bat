@echo off
cd /d "%~dp0"
npx --yes @marp-team/marp-cli@latest --image png --allow-local-files -o _one.png _one.md
