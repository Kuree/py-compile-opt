import argparse
import re
import os
import sysconfig

from typing import Dict, Union


class PycOp:
    def __init__(self, raw_name: str, value: int):
        self.raw_name = raw_name
        self.value = value

        self.op_name = "Pyc_" + "".join([s.capitalize() for s in raw_name.split("_")]) + "Op"
        self.asm_name = raw_name.lower()

        self.num_pushed: Union[int, str] = ""
        self.num_popped: Union[int, str] = ""

    @property
    def stack_unchanged(self):
        return self.num_pushed == self.num_popped

    @property
    def push_stack(self):
        return self.num_pushed

    @property
    def pop_stack(self):
        return self.num_popped

    @property
    def constant_push(self):
        return isinstance(self.num_pushed, int)

    @property
    def constant_pop(self):
        return isinstance(self.num_popped, int)

    def __repr__(self):
        return str({"name": self.asm_name, "value": self.value, "pop": self.num_popped, "push": self.num_pushed})


def parse_opt_defs(include_path: str) -> Dict[str, PycOp]:
    filename = os.path.join(include_path, "opcode.h")
    opcodes = set()
    with open(filename, "r") as f:
        content = f.read()
    if "CACHE" not in content:
        # newer python. need to read out opcode_ids.h
        filename = os.path.join(include_path, "opcode_ids.h")
        with open(filename, "r") as f:
            content = f.read()
    regex = r"#define ([A-Z_]+)\s+(\d+)"
    result: Dict[str, PycOp] = {}
    matches = re.finditer(regex, content, re.MULTILINE)
    for match in matches:
        name: str = match.group(1)
        value: int = int(match.group(2))
        # no repeat
        if value in opcodes:
            continue
        opcodes.add(value)
        # not interested in NB_
        if name.startswith("NB_"):
            continue

        result[name] = PycOp(name, value)

    return result


def parse_opt_nums(op_defs: Dict[str, PycOp]):
    filename = os.path.join(os.path.dirname(__file__), "pycore_opcode_metadata.h")
    with open(filename, "r") as f:
        content = f.read()
    # split based on the
    functions = re.split(r"int _\w+\(int opcode, int oparg\)\s+\{", content)
    assert (len(functions)) == 3
    pop_defs = functions[1]
    push_defs = functions[2]

    matches = re.finditer(r"case ([A-Z_]+):\n\s+return\s(.*);", pop_defs, re.MULTILINE)
    for match in matches:
        name: str = match.group(1)
        value: Union[str, int] = match.group(2)
        if name not in op_defs:
            continue
        op = op_defs[name]
        if value.isdigit():
            value = int(value)
        op.num_pushed = value

    matches = re.finditer(r"case ([A-Z_]+):\n\s+return\s(.*);", push_defs, re.MULTILINE)
    for match in matches:
        name: str = match.group(1)
        value: Union[str, int] = match.group(2)
        if name not in op_defs:
            continue
        op = op_defs[name]
        if value.isdigit():
            value = int(value)
        op.num_pushed = value


def generate_td(filename: str, op_defs: Dict[str, PycOp]):
    content = ""

    for _, op_def in op_defs.items():
        content += "def " + op_def.op_name+ " : Pyc_CodeOp<\"" + op_def.asm_name + "\", ["
        # insert traits
        traits = []

        if op_def.constant_pop:
            traits.append("ConstantPop")
        if op_def.constant_push:
            traits.append("ConstantPush")
        if op_def.pop_stack:
            traits.append("PopStack")
        if op_def.push_stack:
            traits.append("PushStack")
        if op_def.stack_unchanged:
            traits.append("StackUnchanged")
        # TODO: add back native trait implementation
        traits = []
        if len(traits) > 0:
            content += ", ".join(traits) + ", "
        # stack interface
        content += "StackInterface]> {"

        # op interface
        if not op_def.pop_stack:
            pop_code = "return 0;"
        elif isinstance(op_def.pop_stack, int):
            pop_code = f"return {op_def.pop_stack};"
        else:
            pop_code = f"int oparg = getOpArg().getInt(); return {op_def.pop_stack};"

        if not op_def.push_stack:
            push_code = "return 0;"
        elif isinstance(op_def.push_stack, int):
            push_code = f"return {op_def.push_stack};"
        else:
            push_code = f"int oparg = getOpArg().getInt(); return {op_def.push_stack};"
        content += f"""
  let extraClassDeclaration = [{{
    int32_t getNumOfStackPopped() {{ {pop_code} }}
    int32_t getNumOfStackPushed() {{ {push_code} }}
  }}];
"""

        content += "}\n\n"

    with open(filename, "w+") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser("Generate MLIR tablegen for for pyc ops")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    include_path = sysconfig.get_path("include")
    op_defs = parse_opt_defs(include_path)
    parse_opt_nums(op_defs)
    generate_td(args.output, op_defs)


if __name__ == "__main__":
    main()
