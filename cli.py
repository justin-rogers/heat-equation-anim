import argparse
import os
import sys
import heat_eqn_adi
from math import cos, pi


def initial_function(x, y):
    """Initial data, can be modified at will"""
    r2 = ((x - 0.5)**2 + (y - 0.5)**2)
    return abs(cos(r2 * 10 * pi))


graph_parser = argparse.ArgumentParser(
    description='Animate a solution to a heat equation.',
    epilog=
    'The initial function can be modified in cli.py. Example usage: python3 cli.py 50 3000',
)
graph_parser.add_argument(
    'rdx',
    type=int,
    metavar='spatial-resolution',
    help='(int > 2) number of grid points on each axis, higher is finer.')
graph_parser.add_argument(
    'rdt',
    type=int,
    metavar='time-resolution',
    help='(int) number of time steps, a large number yields a gradual animation'
)
graph_parser.add_argument(
    '--title',
    metavar='save-title',
    type=str,
    help='if provided, saves animation as save-title.gif')

args = graph_parser.parse_args()
dx, dt = 1 / args.rdx, 1 / args.rdt
title = args.title

# save animation?
save = False
if title != None:
    save = True

y = heat_eqn_adi.HeatEqnSolver(dx=dx, dt=dt, init_data=initial_function, D=1)
y.anim(
    1,  # iterations per frame
    60,  # frames
    delay=100,  # milliseconds of delay between frames
    save=save,
    title=title)
