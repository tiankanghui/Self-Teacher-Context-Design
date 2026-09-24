"""Trusted answer targets for the seven proof rows with an empty Answer field.

This module intentionally has no third-party dependencies so generation jobs
can import it from lightweight vLLM serving images.
"""

PROOF_ANSWER_OVERRIDES = {
    "221514d62e08cf84ccb66633580113122ac0bec6a8e256c373603bfb40eb0a5d": (
        r"\text{The sum of the polygon's interior angles is divisible by }360^\circ."
    ),
    "731970094ff9363491bbca029771b446929af5fa8c7a5f314f76aba13a69bb4f": (
        r"A\text{ is an infinite set.}"
    ),
    "81022c61f7e58888b092cd3bcd98e68c85530a69e6da7ed6fc7578b048a172e0": (
        r"\frac{a_1}{a_2}+\frac{a_2}{a_3}+\cdots+\frac{a_n}{a_1}\ge n."
    ),
    "7b8e9c20a5d4cd59040c73b1e73e4d5a318ccbc5d54a29bf9214cc2dddc8ddba": (
        r"A,B,C,D\text{ are concyclic.}"
    ),
    "6d6a28445a9be1ad15d328db5f5f936d80ac6f0cb10882f5a6ccebcd5a14c8de": (
        r"(a+b+c)\left(\frac1a+\frac1b+\frac1c\right)\ge 9."
    ),
    "b958e3db6bb3041fd98cd7f3eec3a0c9653c9a996d4ca3559203b2cb19de7227": (
        r"P(x)=a_n(x-x_0)^n."
    ),
    "0549c47572a46d8ac56774e2ec09fc182cb631535dadca78163f3ead354926de": (
        r"\sum_{i=1}^{n}a_i b_i\le\sum_{i=1}^{n}a_i B_i."
    ),
}
