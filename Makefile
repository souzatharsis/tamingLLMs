
build:
	poetry run jupyter-book clean tamingllms/
	poetry run jupyter-book build tamingllms/

build-latex-quick:
	cd tamingllms/latex && pdflatex -shell-escape main.tex

build-latex:
	cd tamingllms/latex && \
	pdflatex -shell-escape main.tex && \
	biber main && \
	pdflatex -shell-escape main.tex && \
	pdflatex -shell-escape main.tex && \
	makeindex main.nlo -s nomencl.ist -o main.nls && \
	pdflatex -shell-escape main.tex > main.log && \
	cd ../..

clean:
	poetry run jupyter-book clean tamingllms/


convert-to-markdown:
	poetry run jupyter nbconvert --to markdown $(file)

d2:
	d2 -t 1 --sketch $(file) $(output)



