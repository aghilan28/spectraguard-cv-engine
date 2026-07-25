.PHONY: setup test clean

setup:
pip install -r requirements.txt

test:
pytest tests/ -v

clean:
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
