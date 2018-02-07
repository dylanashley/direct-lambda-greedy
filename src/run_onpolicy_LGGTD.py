#!/usr/bin/env python -O
# -*- coding: ascii -*-

import argparse
import numpy as np
import os
import signal
import sys

from domains import *
from learners import *


def main(args):
    global siginfo_message

    all_rmse = np.ones((args['num_seeds'], args['num_steps'])) * np.nan
    all_lambda = np.copy(all_rmse)

    for seed in range(args['num_seeds']):

        # build domain
        domain = Ringworld.Ringworld(
            args['num_states'], random_generator=np.random.RandomState(seed))
        last_x, action, reward, gamma, x = next(domain)
        last_x = domain.state_to_features(last_x)
        x = domain.state_to_features(x)

        # build learners
        learner = LGGTD.LGGTD(last_x, domain.MAX_GAMMA, domain.MAX_REWARD)

        for step in range(args['num_steps']):

            # set message for siginfo
            siginfo_message = '[{0:3.2f}%] SEED: {1} of {2}, EPISODE: {3} of {4}, STEP: {5}'.format(
                100 * ((seed + step / args['num_steps']) / args['num_seeds']),
                seed + 1, args['num_seeds'], step + 1, args['num_steps'], step)

            # process step
            lambda_ = learner.update(reward, gamma, x, args['alpha'],
                                     args['eta'])

            # record rmse and lambda
            all_rmse[seed, step] = domain.rmse(learner)
            all_lambda[seed, step] = lambda_

            # move to next step
            last_x, action, reward, gamma, x = next(domain)
            last_x = domain.state_to_features(last_x)
            x = domain.state_to_features(x)

    with open('{}/rmse.npz'.format(args['directory']), 'wb') as outfile:
        np.save(outfile, all_rmse)
    with open('{}/lambda.npz'.format(args['directory']), 'wb') as outfile:
        np.save(outfile, all_lambda)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('alpha', type=float)
    parser.add_argument('eta', type=float)
    parser.add_argument('--numseeds', type=int, dest='num_seeds', default=100)
    parser.add_argument('--numstates', type=int, dest='num_states', default=25)
    parser.add_argument('--numsteps', type=int, dest='num_steps')
    args = vars(parser.parse_args())
    if 'num_steps' not in args:
        args['num_steps'] = args['num_states'] * 100
    return args


if __name__ == '__main__':
    # get command line arguments
    args = parse_args()

    # setup numpy
    np.seterr(divide='raise', over='raise', under='ignore', invalid='raise')

    # setup siginfo response system
    global siginfo_message
    siginfo_message = None
    if hasattr(signal, 'SIGINFO'):
        signal.signal(
            signal.SIGINFO,
            lambda signum, frame: sys.stderr.write('{}\n'.format(siginfo_message))
        )

    # parse args and run
    rmse_filename = '{}/rmse.npz'.format(args['directory'])
    lambda_filename = '{}/lambda.npz'.format(args['directory'])
    if not (os.path.exists(rmse_filename) and
            (os.path.exists(lambda_filename))):
        main(args)
