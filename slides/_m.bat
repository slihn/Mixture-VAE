@echo off
cd /d "%~dp0"
call marp --pdf --pdf-outlines --allow-local-files -o frac_slides.pdf frac_slides.md
