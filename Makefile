
build:
	poetry run jupyter-book clean tamingllms/
	poetry run jupyter-book build tamingllms/

build-latex:
	poetry run jupyter-book build tamingllms/latex --builder latex

clean:
	poetry run jupyter-book clean tamingllms/


convert:
	poetry run jupyter nbconvert --to markdown $(file)

d2:
	d2 -t 1 --sketch tamingllms/_static/safety/design.d2 tamingllms/_static/safety/design.svg


convert-latex:
	jupyter nbconvert tamingllms/notebooks/structured_output.ipynb --to latex

