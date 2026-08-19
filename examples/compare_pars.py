#!/usr/bin/env python3

import sys
from pyharm.parameters import parse_parthenon_dat
from configparser import ConfigParser

def compare(pars1, pars2):
    set1 = set(pars1.items())
    set2 = set(pars2.items())
    diff = set1 ^ set2
    if len(diff) > 0:
        print(diff)

def clean(pars, cpars):
    for key in list(pars.keys()):
        typ = type(pars[key])

        if typ == dict:
            if cpars is not None:
                if key in cpars:
                    compare(pars[key], cpars[key])
                else:
                    print("NOT PRESENT: ", key)
            del pars[key]
        elif typ == list:
            if cpars is not None and key in cpars:
                diff = set(pars[key]) ^ set(cpars[key])
                if len(diff) > 0:
                    print(diff)
            del pars[key]
        elif typ == ConfigParser:
            del pars[key]

pars1 = parse_parthenon_dat(open(sys.argv[1]).read())
pars2 = parse_parthenon_dat(open(sys.argv[2]).read())

clean(pars1, pars2)
clean(pars2, None)

compare(pars1, pars2)
