def gate_dict():
    gates = [    
    # one qubit gates
    "id",
    "u1",
    "u2",
    "u3",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "cx",
    "sx",
    # two qubit gates
    "cx",
    "cy",
    "cz",
    "ch",
    "crz",
    "cu1",
    "cu3",
    "swap",
    "rzz"]
    gate_dict = {}
    for i, gate in enumerate(gates):
        gate_dict[gate] = i
    return gate_dict
