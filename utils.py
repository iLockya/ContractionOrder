import numpy as np
import torch


def str_symmetric_difference(str1, str2) -> str:
    remains = set(str1) ^ set(str2)
    result = ""
    for s in str1+str2:
        if s in remains:
            result += s
    return result



if __name__ == "__main__":
    s1 = "abcd"
    s2 = "berd"
    print(str_symmetric_difference(s1,s2))