# Contributing Guide

Thank you for your interest in contributing to the Groningen University AI Compute Depot!

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/load-llm.git
   cd load-llm
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install streamlit httpx cloudpickle
   ```
4. Install the pre-push deploy hook (optional, for maintainers with server access):
   ```bash
   bash deploy.sh --install-hook
   ```

## Making Changes

- Create a branch for your change:
  ```bash
  git checkout -b feature/my-feature
  ```
- Keep commits focused and use clear commit messages
- Test your changes by running the app locally:
  ```bash
  streamlit run app.py
  ```

## Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/my-feature
   ```
2. Open a pull request against the `main` branch
3. Describe what the change does and why
4. Reference any related issues

## Guidelines

- Follow the existing code style — keep it simple and readable
- Do not commit secrets, tokens, or credentials
- Update `CHANGELOG.md` under `[Unreleased]` for any notable changes
- Be respectful — see our [Code of Conduct](CODE_OF_CONDUCT.md)

## Reporting Bugs

Open a [GitHub issue](https://github.com/JvD007/load-llm/issues) with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behaviour

For security vulnerabilities, follow the [Security Policy](SECURITY.md) instead.
