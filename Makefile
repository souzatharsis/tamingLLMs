
build:
	poetry run jupyter-book clean tamingllms/
	poetry run jupyter-book build tamingllms/

clean:
	poetry run jupyter-book clean tamingllms/


convert:
	poetry run jupyter nbconvert --to markdown $(file)

d2:
	d2 -t 1 --sketch tamingllms/_static/safety/design.d2 tamingllms/_static/safety/design.svg
