import argparse
import tkinter as tk
from pec import Analyse_PEC
from pec import DotViewer

parser = argparse.ArgumentParser()
parser.add_argument('--viewer', action='store_true', default=False, required=False,
                    help='Dot viewer')

args = parser.parse_args()

if args.viewer:
    root = tk.Tk()
    app = DotViewer.Viewer(master=root)
    app.start()
else:
    root = tk.Tk()
    app = Analyse_PEC.Analyse_PEC(master=root)
    app.start()