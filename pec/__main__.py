import argparse
from pec.Analyse_PEC import Tk, Analyse_PEC
from pec import DotViewer

parser = argparse.ArgumentParser()
parser.add_argument('--viewer', action='store_true', default=False, required=False,
                    help='Dot viewer')

args = parser.parse_args()

if args.viewer:
    root = Tk()
    app = DotViewer.Viewer(master=root)
    app.start()
else:
    root = Tk()
    app = Analyse_PEC(master=root)
    app.start()