"""
Duplex transport. Barge-in needs audio flowing both ways continuously —
request/response HTTP structurally cannot do it (plan p5, module #2).
Proctor frames never share this socket.
"""
