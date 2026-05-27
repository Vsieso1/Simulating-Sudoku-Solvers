#!/usr/bin/env python
"""
Launch the Sudoku GUI application.

Usage:
    python scripts/run_gui.py
"""

import sys
import tkinter as tk
from pathlib import Path

# Add parent directory to path to import Sudoku module
sys.path.insert(0, str(Path(__file__).parent.parent))

from Sudoku.GUI_Sudoku.GUI import SudokuApp


def main():
    """Launch the GUI application."""
    print("🎮 Launching Sudoku GUI...")
    root = tk.Tk()
    app = SudokuApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
