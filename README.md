# Project Template

`project-temp` – this is my project template that enhances development experience in VSCode with streamlined features.

## Features

- Accessing the project root as a constant.
- Load Parameters from the `.env` file
    - In debug mode, parameters are loaded automatically.
    - Running in the terminal mode need user settings `"python.experiments.optInto": ["pythonTerminalEnvVarActivation"]`
- Jupyter Settings
    - Run Jupyter notebooks from the project root.
    - Enable the interactive mode for development.
- Python
    - Autopep8 formatter
- Plot utils 
- Databse utils

# Reference

- [VS Code Python Environments Documentation](https://code.visualstudio.com/docs/python/environments#_creating-environments)
- [VS Code Python Issue #944](https://github.com/microsoft/vscode-python/issues/944)