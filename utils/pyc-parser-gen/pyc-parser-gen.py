import argparse
import httplib2
import sys
import re

from typing import Dict


def fetch_marshal() -> str:
    # use the sys info to decide which version to download
    major = sys.version_info.major
    minor = sys.version_info.minor
    url = f"https://raw.githubusercontent.com/python/cpython/refs/heads/{major}.{minor}/Python/marshal.c"
    http = httplib2.Http()
    header, data = http.request(url)
    return data.decode("ascii")


def generate_logic(content: str, filename: str):
    # first extract out type def
    regex = r"#define ([A-Z_]+)\s+'(.+)'"
    matches = re.finditer(regex, content, re.MULTILINE)
    enums = []
    for match in matches:
        name: str = match.group(1)
        value: str = match.group(2)
        if name.startswith("TYPE"):
            enums.append((name, value))

    # figure out the code object format
    regex = r"w_([a-z]+)\((co->|)co_(.+),"
    matches = re.finditer(regex, content, re.MULTILINE)
    code_objs = []
    for match in matches:
        type_str = match.groups()[0]
        var_name = match.groups()[-1]
        code_objs.append((type_str, var_name))

    # generate code
    content = """
enum class ObjectType: int8_t {    
"""
    for name, v in enums:
        content += f"    {name} = '{v}',\n"
    content += """
};

"""

    # generate code object code
    for type_str, name in code_objs:
        content += f"#define PYC_{name.upper()}\n"

    with open(filename, "w+") as f:
        f.write(content)

def main():
    arg_parser = argparse.ArgumentParser("pyc parser logic generator")
    arg_parser.add_argument("-o", "--output")
    args = arg_parser.parse_args()

    content = fetch_marshal()
    generate_logic(content, args.output)

if __name__ == "__main__":
    main()
