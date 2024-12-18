# MemeCraft

The following is a repository for the final project for Transfromers in CV course.

# Usage

## Set up the environment

Run the following command in your virtual environment to install Poetry using the official installer:
`pipx install poetry`

Set your local configuration with: 
`poetry config virtualenvs.create false --local`

Install all necessary dependencies from pyproject.toml:
`poetry install`

## Proceed with the following

- Download additional datasets using `get_dataset.sh` if you wish.
- Run `eval_embeddings.py` to update embeddings.
    ```
    python3 eval_embeddings.py
    ```
- Run `main.py` to generate memes
    ```
    python3 main.py
    ```
    Use flag `-p` to pass your own meme caption, flag `-t` to pass a topic for a caption, and flag `-d` to specify path to the dataset.

# Team Members

- Boris Mikheev
- Ivan Listopadov
- Roman Makarov
- Sergey Grozny
- Yurii Melnik
