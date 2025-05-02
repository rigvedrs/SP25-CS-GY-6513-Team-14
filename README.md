# Project Environment Setup


## Setting Up the Environment

### 1. Install ASDF
If you don't have ASDF installed, [Read the installation guide](asdf.md)

### 2. Install Required Plugins
```bash
asdf plugin add python
asdf plugin add java
```

### 3. Install Runtime Versions
With the .tool-versions file in your project root, simply run:
```bash
asdf install
```

### 4. Python Dependencies
After the Python version is installed and activated, install the dependencies:
```bash
pip install -r requirements.txt
```

### 5. Virtual Environment (Optional)
If using a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verification
To verify your setup is correct:
```bash
asdf current
which python
python --version
which java
java -version
```

## Notes
- The requirements.txt file contains all Python packages needed for this project
- Make sure ASDF is properly loaded in your shell before running any commands