from setuptools import setup, find_packages

setup(
    name="crypto-bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pyTelegramBotAPI==4.14.1",
        "requests==2.31.0",
        "numpy==1.24.3",
        "pandas==2.0.3",
    ],
    python_requires=">=3.8, <3.12",
)
