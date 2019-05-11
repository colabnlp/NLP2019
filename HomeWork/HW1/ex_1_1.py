import pynini as pn
import random

# compose - *
# concat  - +
# union   - |


fst = (pn.a("a") | pn.a("e")) + pn.t("a", pn.a("0").closure(0,5)) | pn.t(pn.a("a").star, "0") + pn.a("xxx")
fst = fst.optimize()

output_strings = set()

for i in range(10000):
    s = pn.randgen(fst,1,random.randint(0,100000)).stringify()
    output_strings.add(s)


print(len(output_strings))

for output_string in output_strings:
    print(output_string)


def top_paths(fst, count=100):
    return sorted(set(p[1] for p in pn.shortestpath(fst, nshortest=count).paths()))


print("INPUTS")
print("\t")
print(*top_paths(pn.invert(fst), 20), sep="\n\t")

