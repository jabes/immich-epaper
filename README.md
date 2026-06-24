### Local Development

```bash
pyenv install 3.12 --skip-existing
pyenv local 3.12
pyenv deactivate
pyenv virtualenv-delete --force immich-epaper
pyenv virtualenv 3.12 immich-epaper
pyenv activate immich-epaper
pip install --upgrade wheel setuptools pip
pip install --requirement requirements.txt
```
