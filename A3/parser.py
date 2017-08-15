import re
import sys
from collections import defaultdict as ddict
from bn_learning import *
from learning_helpers import *
import os

def writeMarkov(filename, factors):
    """Parses the .bif file with the given name."""

    # Setting up I/O
    module_name = filename+'_mn'
    outfile = open(module_name + '.txt', 'wb')

    def write(s):
        outfile.write(s)

    infile = open(filename+'.bif')

    infile.readline()
    infile.readline()


    # Regex patterns for parsing
    variable_pattern = re.compile(r"  type discrete \[ \d+ \] \{ (.+) \};\s*")
    prior_probability_pattern_1 = re.compile(
        r"probability \( ([^|]+) \) \{\s*")
    prior_probability_pattern_2 = re.compile(r"  table (.+);\s*")
    conditional_probability_pattern_1 = (
        re.compile(r"probability \( (.+) \| (.+) \) \{\s*"))
    conditional_probability_pattern_2 = re.compile(r"  \((.+)\) (.+);\s*")

    variables = {}  # domains
    tree = {}
    cpts = []
    # For every line in the file
    
    while True:
        line = infile.readline()

        # End of file
        if not line:
            break

        # Variable declaration
        if line.startswith("variable"):
            write(line)

            curr_line = infile.readline()
            write(curr_line)
            match = variable_pattern.match(curr_line)
            # Extract domain and place into dictionary
            if match:
                variable = line[9:-3]
                tree[variable] = set()
                variables[variable] = match.group(1).split(", ")
            else:
                raise Exception("Unrecognised variable declaration:\n" + line)
            curr_line =infile.readline()
            write(curr_line)


    for factor in factors:
        write(printFormat(factor, variables))


def writeNet(filename, cptDict):
    """Parses the .bif file with the given name."""

    # Setting up I/O
    module_name = filename+'_bn'
    outfile = open(module_name + '.bif', 'wb')

    def write(s):
        outfile.write(s)

    infile = open(filename+'.bif')

    curr_line = infile.readline()
    write(curr_line)
    curr_line = infile.readline()
    write(curr_line)

    # Regex patterns for parsing
    variable_pattern = re.compile(r"  type discrete \[ \d+ \] \{ (.+) \};\s*")
    prior_probability_pattern_1 = re.compile(
        r"probability \( ([^|]+) \) \{\s*")
    prior_probability_pattern_2 = re.compile(r"  table (.+);\s*")
    conditional_probability_pattern_1 = (
        re.compile(r"probability \( (.+) \| (.+) \) \{\s*"))
    conditional_probability_pattern_2 = re.compile(r"  \((.+)\) (.+);\s*")

    variables = {}  # domains
    tree = {}
    cpts = []
    # For every line in the file
    
    while True:
        line = infile.readline()
        write(line)

        # End of file
        if not line:
            break

        # Variable declaration
        if line.startswith("variable"):
            curr_line = infile.readline()
            write(curr_line)
            match = variable_pattern.match(curr_line)
            # Extract domain and place into dictionary
            if match:
                variable = line[9:-3]
                tree[variable] = set()
                variables[variable] = match.group(1).split(", ")
            else:
                raise Exception("Unrecognised variable declaration:\n" + line)
            curr_line =infile.readline()
            write(curr_line)

        # Probability distribution
        elif line.startswith("probability"):

            match = prior_probability_pattern_1.match(line)
            if match:
                # Prior probabilities
                variable = match.group(1)
                line = infile.readline()
                probabs = []
                for val in variables[variable]:
                    probabs.append(str(cptDict[variable].prior_count[val]))

                probabs = ", ".join(probabs)
                line = "  table " + probabs + ";\n"
                write(line)
               # print(line)
                match = prior_probability_pattern_2.match(line)
                currCPT = CPT(variable, variables[variable], [], [])
                cpts.append(currCPT)

                curr_line = infile.readline()  # }
                write(curr_line)
            else:
                match = conditional_probability_pattern_1.match(line)
                if match:
                    # Conditional probabilities
                    variable = match.group(1)
                    given = match.group(2).split(", ")
                    evidence = set()
                    # Iterate through the conditional probability table
                    while True:
                        line = infile.readline()  # line of the CPT
                        if line == '}\n':
                            write(line)
                            break

                        match = conditional_probability_pattern_2.match(line)
                        given_values = match.group(1)
                        start_line = "  (" + given_values + ") "
                        given_values = tuple(given_values.split(", "))
                        evidence.add(given_values)
                        probabs = []
                        for val in variables[variable]:
                            probabs.append(str(cptDict[variable].table[given_values][val]))
                        
                        probabs = ", ".join(probabs)
                        curr_line = start_line + " " + probabs + ";\n"
                        write(curr_line)
                       # print(curr_line)

                    for evidenceVar in given:
                        tree[evidenceVar].add(variable)
                        
                    currCPT = CPT(variable, variables[variable], evidence, given)
                    cpts.append(currCPT)
        
                else:
                    raise Exception("Unrecognised probability declaration:\n" + line)


    outfile.close()
    return tree, cpts, variables


def parse(filename, trained = False):
    """Parses the .bif file with the given name."""

    # Setting up I/O

    infile = open(filename+'.bif')

    infile.readline()
    infile.readline()


    # Regex patterns for parsing
    variable_pattern = re.compile(r"  type discrete \[ \d+ \] \{ (.+) \};\s*")
    prior_probability_pattern_1 = re.compile(
        r"probability \( ([^|]+) \) \{\s*")
    prior_probability_pattern_2 = re.compile(r"  table (.+);\s*")
    conditional_probability_pattern_1 = (
        re.compile(r"probability \( (.+) \| (.+) \) \{\s*"))
    conditional_probability_pattern_2 = re.compile(r"  \((.+)\) (.+);\s*")

    variables = {}  # domains
    tree = {}
    cpts = []
    # For every line in the file

    while True:
        line = infile.readline()

        # End of file
        if not line:
            break

        # Variable declaration
        if line.startswith("variable"):
            match = variable_pattern.match(infile.readline())
            # Extract domain and place into dictionary
            if match:
                variable = line[9:-3]
                tree[variable] = set()
                variables[variable] = match.group(1).split(", ")
            else:
                raise Exception("Unrecognised variable declaration:\n" + line)
            infile.readline()

        # Probability distribution
        elif line.startswith("probability"):

            match = prior_probability_pattern_1.match(line)
            if match:

                # Prior probabilities
                variable = match.group(1)
                line = infile.readline()
                match = prior_probability_pattern_2.match(line)
                currCPT = CPT(variable, variables[variable], [], [])

                if trained:
                    probabilities = map(float, match.group(1).split(", "))
                    for i,var in enumerate(variables[variable]):
                        currCPT.prior_count[var] = probabilities[i]

                cpts.append(currCPT)

                infile.readline()  # }

            else:
                match = conditional_probability_pattern_1.match(line)
                if match:

                    # Conditional probabilities
                    variable = match.group(1)
                    given = match.group(2).split(", ")
                    evidence = set()
                    if trained:
                        probabilities = {}

                    # Iterate through the conditional probability table
                    while True:
                        line = infile.readline()  # line of the CPT
                        if line == '}\n':
                            break
                        match = conditional_probability_pattern_2.match(line)
                        given_values = match.group(1)
                        given_values = tuple(given_values.split(", "))
                        evidence.add(given_values)
                        if trained:
                            probabilities[given_values] = map(float, match.group(2).split(", "))

                      

                    for evidenceVar in given:
                        tree[evidenceVar].add(variable)
                        
                    currCPT = CPT(variable, variables[variable], evidence, given)

                    if trained:
                        for evidence_assignment in evidence:
                            for i, var in enumerate(variables[variable]):
                                currCPT.table[evidence_assignment][var] = probabilities[evidence_assignment][i]

                    cpts.append(currCPT)
        
                else:
                    raise Exception("Unrecognised probability declaration:\n" + line)


    return tree, cpts, variables


