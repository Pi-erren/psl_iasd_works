**INSTALL POETRY** (https://python-poetry.org/docs/)

brew install pipx  
pipx ensurepath # Ensure pipx is in PATH  
pipx install poetry # Install poetry  
poetry --version # Ensure poetry installed  

-
-
-

**SETUP POETRY PROJECT**  

1) Initialize poetry project and add dependencies  
    poetry init  
    poetry add <package_name>  
    poetry install  OR poetry install --no-root (use poetry only for dependencies management)

2) Manage manually virtual environment
    - activate virutal environment:   
        poetry shell  
    - exite virtual environment:   
        exit

3) Setup Visual Studio Code Python Interpreter (The path of the python environment used by VSC)
	- Use Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (macOS)
	- Search for "Python: Select Interpreter"
	- Select the poetry virtual environment  
	- If the poetry env does not appear:
		- poetry env info --path
		- Copy paste in Python: Select Interpreter" > "Enter interpreter path"
  
4) If you don't want to package your project and only need Poetry for managing dependencies, modify your pyproject.toml to disable the "package mode": 

    - In the terminal:  
	    poetry install --no-root  
    OR
    - In the .toml file  
	    [tool.poetry]  
	    package-mode = false