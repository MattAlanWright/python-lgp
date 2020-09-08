import numpy as np
import copy
import pickle

from utils import weightedCoinFlip


def ConfigureProgram():
    Program.MAX_SOURCE_INDEX = max(Program.NUM_REGISTERS, Program.NUM_INPUTS)


class Program(object):

    ADDITION_OP                = 0
    SUBTRACTION_OP             = 1
    MULTIPLICATION_OP          = 2
    DIVISION_OP                = 3
    COS_OP                     = 4
    CONDITIONAL_OP             = 5
    NUM_OP_CODES               = 6

    OP_SYMBOLS                 = ['+', '-', '*', '/', 'cos', '?']

    MIN_RESULT                 = np.finfo(np.float32).min
    MAX_RESULT                 = np.finfo(np.float32).max
    DEFAULT_OP_RESULT          = np.float32(0.0)

    MODE_INDEX                 = 0
    TARGET_INDEX               = 1
    OP_CODE_INDEX              = 2
    SOURCE_INDEX               = 3
    NUM_INSTRUCTION_COMPONENTS = 4

    REGISTER_MODE              = 0
    INPUT_MODE                 = 1
    NUM_MODES                  = 2

    NUM_REGISTERS               = 8
    NUM_INPUTS                  = 4
    MAX_SOURCE_INDEX            = max(8, 4) # This should be the max of the number of registers and the input size
    MUTATION_RATE               = 0.20

    MIN_PROG_SIZE               = 32
    MAX_PROG_SIZE               = 1024

    P_ADD                       = 0.7
    P_DEL                       = 0.7
    P_MUT                       = 0.8

    def __init__(self):

        # Pre-calculate mod value depending on source access mode
        self._source_mod_value = [0, 0]
        self._source_mod_value[Program.INPUT_MODE]    = Program.NUM_INPUTS
        self._source_mod_value[Program.REGISTER_MODE] = Program.NUM_REGISTERS

        # Allocate space for registers
        self.registers = np.zeros(Program.NUM_REGISTERS)

        # Initialize random instructions.
        self._instructions = []
        num_new_instructions = np.random.randint(Program.MIN_PROG_SIZE, Program.MAX_PROG_SIZE + 1)
        for _ in range(num_new_instructions):
            self._instructions.append(self.createRandomInstruction())


    def createRandomInstruction(self):
        mode          = np.random.randint(0, Program.NUM_MODES)
        target_index  = np.random.randint(0, Program.NUM_REGISTERS)
        op_code       = np.random.randint(0, Program.NUM_OP_CODES)
        source_index  = np.random.randint(0, Program.MAX_SOURCE_INDEX)

        return [mode, target_index, op_code, source_index]


    def printInstructions(self):
        for instruction in self._instructions:
            self.printInstruction(instruction)


    def printInstruction(self, instruction):
        mode, target_index, op_code, source_index = instruction

        source = None
        if mode == Program.REGISTER_MODE:
            source = 'R'
        else:
            source = 'IP'

        source_index %= self._source_mod_value[mode]

        instruction_string = 'R[' + \
                             str(target_index) + \
                             '] <- ' + \
                             'R[' + \
                             str(target_index) + \
                             '] ' + \
                             Program.OP_SYMBOLS[op_code] + \
                             " " + \
                             source + \
                             '[' + \
                             str(source_index) + \
                             ']'

        print(instruction_string)


    def cleanResult(self, result):
        '''The result of instruction execution may be NaN, inf, or -inf. This function
        cleans the resulting value to ensure that it is a valid np.float32 value.
        '''

        result = np.float32(result)
        if np.isnan(result):
            result = np.float32(0.0)
        elif np.isinf(result) and result < 0.0:
            result = Program.MIN_RESULT
        elif np.isinf(result):
            result = Program.MAX_RESULT

        return min(max(result, Program.MIN_RESULT), Program.MAX_RESULT)


    def execute(self, state):
        for instruction in self._instructions:
            self.executeInstruction(instruction, state)


    def executeInstruction(self, instruction, state):
        '''Deconstruct components out of the instruction and perform the
        operation.

        mode:       Indicates whether the source operand is a register or state value
        target:     Destination register and operand
        source:     Register or state value oeprand
        op_code:    Operation to perform
        '''

        mode, target_index, op_code, source_index = instruction

        target = self.registers[target_index]

        source = None
        source_index = source_index % self._source_mod_value[mode]
        if mode == Program.REGISTER_MODE:
            source = self.registers[source_index]
        else:
            source = state[source_index]

        result = 0
        if op_code == Program.ADDITION_OP:
            result = target + source

        elif op_code == Program.SUBTRACTION_OP:
            result = target - source

        elif op_code == Program.MULTIPLICATION_OP:
            result = source * 2.0

        elif op_code == Program.DIVISION_OP:
            result = source / 2.0

        elif op_code == Program.COS_OP:
            result = np.cos(source)

        elif op_code == Program.CONDITIONAL_OP:
            if target < source:
                result = -target

        result = self.cleanResult(result)

        self.registers[target_index] = result


    def deleteRandomInstruction(self):
        index = np.random.randint(0, len(self._instructions))
        self._instructions = np.delete(self._instructions, index, axis=0)


    def addRandomInstruction(self):
        index = np.random.randint(0, len(self._instructions))
        self._instructions = np.insert(self._instructions,
                                      index,
                                      self.createRandomInstruction(),
                                      axis=0)


    def mutateRandomInstruction(self):
        instruction_index = np.random.randint(0, len(self._instructions))
        component_index   = np.random.randint(0, Program.NUM_INSTRUCTION_COMPONENTS)

        upper_bound = None
        if component_index == Program.MODE_INDEX:
            upper_bound = Program.NUM_MODES
        elif component_index == Program.TARGET_INDEX:
            upper_bound = Program.NUM_REGISTERS
        elif component_index == Program.OP_CODE_INDEX:
            upper_bound = Program.NUM_OP_CODES
        else:
            upper_bound = Program.MAX_SOURCE_INDEX

        # Mutate component
        new_val = np.random.randint(0, upper_bound)
        while new_val == self._instructions[instruction_index][component_index]:
            new_val = np.random.randint(0, upper_bound)
        self._instructions[instruction_index][component_index] = new_val


    def mutate(self):

        # Random instruction deletion
        while weightedCoinFlip(Program.P_DEL) and len(self._instructions) > Program.MIN_PROG_SIZE:
            self.deleteRandomInstruction()

        # Random instruction addition (creation)
        while weightedCoinFlip(Program.P_ADD) and len(self._instructions) < Program.MAX_PROG_SIZE:
            self.addRandomInstruction()

        # Random instructions mutation
        while weightedCoinFlip(Program.P_MUT):
            self.mutateRandomInstruction()
