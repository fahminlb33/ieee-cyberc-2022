.PHONY: format

format:
	yapf -i -r -p "experiments/" *.py
