"""
CodeRefactor AI — Automated Python Code Refactoring Pipeline
=============================================================
Analyzes Python repositories for code smells, recommends the best LLM
for refactoring, and executes the refactoring via OpenRouter API.

Pipeline stages:
  1. Scan repository → extract all Python functions
  2. Code smell detection → identify bad-smell functions
  3. LLM recommendation → predict best model for each bad function
  4. Refactoring → call OpenRouter API to refactor
  5. Apply changes → write refactored code back
"""

__version__ = "1.0.0"
