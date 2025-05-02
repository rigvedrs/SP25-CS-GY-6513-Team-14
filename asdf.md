# asdf Installation Guide

## Installation

```bash
brew install asdf
```

## Configuration

Add the following to your shell configuration file (e.g., `.zshrc`, `.bashrc`):

```bash
# Add asdf to PATH
export PATH="${ASDF_DATA_DIR:-$HOME/.asdf}/shims:$PATH"

# Set up completions for zsh
mkdir -p "${ASDF_DATA_DIR:-$HOME/.asdf}/completions"
asdf completion zsh > "${ASDF_DATA_DIR:-$HOME/.asdf}/completions/_asdf"

# Append completions to fpath
fpath=(${ASDF_DATA_DIR:-$HOME/.asdf}/completions $fpath)

# Initialize completions with ZSH's compinit
autoload -Uz compinit && compinit
```

After adding these lines, restart your terminal or run `source ~/.zshrc` (or your shell config file).