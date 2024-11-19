
build:
	rm -rf tamingllms/_build/html/*
	poetry run jupyter-book build tamingllms/

clean:
	poetry run jupyter-book clean tamingllms/