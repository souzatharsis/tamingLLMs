
build:
	poetry run jupyter-book clean tamingllms/
	poetry run jupyter-book build tamingllms/

clean:
	poetry run jupyter-book clean tamingllms/


convert:
	poetry run jupyter nbconvert --to markdown $(file)