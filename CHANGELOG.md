# Changelog

All notable changes to RAGLab are documented here.
This file is generated from [conventional commits](https://www.conventionalcommits.org) by [git-cliff](https://git-cliff.org).

## [Unreleased]

### Bug Fixes

- Update project directory path in README for clarity
- Handle missing LangChain imports with stubs, improve error messaging, and add prompt logging to generation and .env persistence to GUI

### Chores

- Update .gitignore and configuration for environment variable loading; change notebook kernel display name
- Migrate dependency management to uv and simplify chunking strategy documentation
- Replace hardcoded local file paths with generic placeholders in MCP_SETUP.md

### Features

- Implement initial Retrieval Augmented Generation (RAG) system including data loading, chunking, vector store, retrieval, generation, and testing infrastructure.
- Implement Streamlit GUI for collection and RAG management
- Add multi-collection support, query rewriting, and a GUI application while removing legacy data loaders and documentation.
- Add Docker image, CI/CD, and automated GHCR releases

### Refactoring

- Modernize type hinting by replacing typing collections with built-in equivalents and clean up imports
