#!/bin/bash
# Встановлюємо Python 3.11 через pyenv
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install 3.11.9
pyenv global 3.11.9

# Встановлюємо залежності
pip install --upgrade pip
pip install -r requirements.txt