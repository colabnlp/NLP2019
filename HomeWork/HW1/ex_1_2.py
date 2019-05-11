import pynini as pn
from num2words import num2words
import random

# https://tuto.pages-informatique.com/writing-out-numbers-in-french-letters.php
# http://www.woodwardfrench.com/lesson/numbers-from-1-to-100-in-french/
# http://www.heartandcoeur.com/convert/convert_chiffre_lettre.php

# compose - *
# concat  - +
# union   - |


def generate_fst_for_factor_digit(factor, include_zero=False):
    fst = pn.Fst()
    carets = ''
    if factor > 0:
        carets = '^' * factor
        carets = carets + ' '

    for num in range(0, 10):

        # if num == 0 and include_zero is False:
        #     fst_temp = pn.t(str(num), "")
        # else:
        #     fst_temp = pn.t(str(num), str(num) + carets)
        fst_temp = pn.t(str(num), str(num) + carets)
        fst = pn.union(fst, fst_temp)

    fst = fst.optimize()

    return fst


def generate_fst_digit():
    fst_dict = {}

    fst_single_digit = generate_fst_for_factor_digit(0, True)

    for factor in range(0, 10):
        fst_dict[factor] = generate_fst_for_factor_digit(factor)

    fst = pn.a("")
    for num_places in range(1, 10):
        fst_for_x_digit_num = pn.a("")

        for num_place in range(num_places, 0, -1):
            if num_places == 1:
                fst_for_x_digit_num = pn.concat(fst_for_x_digit_num, fst_single_digit)
            else:
                fst_for_x_digit_num = pn.concat(fst_for_x_digit_num, fst_dict[num_place - 1])

        fst = pn.union(fst, fst_for_x_digit_num)

    comma_numbers = pn.u(".", ",") + pn.u(*"0123456789").star
    fst = fst + pn.u("", comma_numbers)
    fst = fst.optimize()
    return fst


def generate_french_numerals_hyphens(factor_fst):
    # full french alphabet - https://en.wikiversity.org/wiki/French/Alphabet
    alphabet_full = pn.u(*".0123456789^ -abcdefghijklmnopqrstuvwxyzàèùéâêîôûëïüÿæœç").star
    fsa_0_9 = pn.u(*"0123456789").star

    single_zero = pn.t("0", "zéro")

    single_digits = pn.string_map({
        "0": "",  # zéro
        "1": "un",
        "2": "deux",
        "3": "trois",
        "4": "quatre",
        "5": "cinq",
        "6": "six",
        "7": "sept",
        "8": "huit",
        "9": "neuf",
    })

    zeros = pn.string_map({
        # "0^^ 0^ 0": "",
        "0^ ":  "",
        "0^^ ": "",
        "mille-0^^ 0^ 0":   "mille",
    })

    teens_10_19 = pn.string_map({
        "1^ 0": "dix",
        "1^ 1": "onze",
        "1^ 2": "douze",
        "1^ 3": "treize",
        "1^ 4": "quatorze",
        "1^ 5": "quinze",
        "1^ 6": "seize",
        "1^ 7": "dix-sept",
        "1^ 8": "dix-huit",
        "1^ 9": "dix-neuf",
    })

    mult_20_60 = pn.string_map({
        "2^ 0": "vingt",
        "2^ 1": "vingt-et-un",
        "3^ 0": "trente",
        "3^ 1": "trente-et-un",
        "4^ 0": "quarante",
        "4^ 1": "quarante-et-un",
        "5^ 0": "cinquante",
        "5^ 1": "cinquante-et-un",
        "6^ 0": "soixante",
        "6^ 1": "soixante-et-un",
    })

    mult_2x_6x = pn.string_map({
        "2^ ": "vingt-",
        "3^ ": "trente-",
        "4^ ": "quarante-",
        "5^ ": "cinquante-",
        "6^ ": "soixante-",
    })

    mult_70_90 = pn.string_map({
        "7^ 0": "soixante-dix",
        "7^ 1": "soixante-et-onze",
        "7^ 2": "soixante-douze",
        "7^ 3": "soixante-treize",
        "7^ 4": "soixante-quatorze",
        "7^ 5": "soixante-quinze",
        "7^ 6": "soixante-seize",
        "7^ 7": "soixante-dix-sept",
        "7^ 8": "soixante-dix-huit",
        "7^ 9": "soixante-dix-neuf",

        "8^ 0": "quatre-vingts",

        "9^ 0": "quatre-vingt-dix",
        "9^ 1": "quatre-vingt-onze",
        "9^ 2": "quatre-vingt-douze",
        "9^ 3": "quatre-vingt-treize",
        "9^ 4": "quatre-vingt-quatorze",
        "9^ 5": "quatre-vingt-quinze",
        "9^ 6": "quatre-vingt-seize",
        "9^ 7": "quatre-vingt-dix-sept",
        "9^ 8": "quatre-vingt-dix-huit",
        "9^ 9": "quatre-vingt-dix-neuf",
    })

    mult_8x = pn.string_map({
        "8^ ": "quatre-vingt-",
    })

    hundreds_alone = pn.string_map({
        "1^^ 0^ 0": "cent",
        "2^^ 0^ 0": "deux-cents",
        "3^^ 0^ 0": "trois-cents",
        "4^^ 0^ 0": "quatre-cents",
        "5^^ 0^ 0": "cinq-cents",
        "6^^ 0^ 0": "six-cents",
        "7^^ 0^ 0": "sept-cents",
        "8^^ 0^ 0": "huit-cents",
        "9^^ 0^ 0": "neuf-cents",
    })

    hundreds = pn.string_map({
        "1^^ ": "cent-",
        "2^^ ": "deux-cent-",
        "3^^ ": "trois-cent-",
        "4^^ ": "quatre-cent-",
        "5^^ ": "cinq-cent-",
        "6^^ ": "six-cent-",
        "7^^ ": "sept-cent-",
        "8^^ ": "huit-cent-",
        "9^^ ": "neuf-cent-",
    })

    mille = pn.string_map({
        "0^^^ ": "0^^^-mille-",
        "1^^^ ": "1^^^-mille-",
        "2^^^ ": "2^^^-mille-",
        "3^^^ ": "3^^^-mille-",
        "4^^^ ": "4^^^-mille-",
        "5^^^ ": "5^^^-mille-",
        "6^^^ ": "6^^^-mille-",
        "7^^^ ": "7^^^-mille-",
        "8^^^ ": "8^^^-mille-",
        "9^^^ ": "9^^^-mille-",
    })

    million = pn.string_map({
        "0^^^^^^ ": "0^^^^^^-millions-",
        "1^^^^^^ ": "1^^^^^^-millions-",
        "2^^^^^^ ": "2^^^^^^-millions-",
        "3^^^^^^ ": "3^^^^^^-millions-",
        "4^^^^^^ ": "4^^^^^^-millions-",
        "5^^^^^^ ": "5^^^^^^-millions-",
        "6^^^^^^ ": "6^^^^^^-millions-",
        "7^^^^^^ ": "7^^^^^^-millions-",
        "8^^^^^^ ": "8^^^^^^-millions-",
        "9^^^^^^ ": "9^^^^^^-millions-",
    })

    strip_triple_factor = pn.string_map({
        "^^^^^^^^": "^^",
        "^^^^^^^": "^",
        "^^^^^^": "",
        "^^^^^": "^^",
        "^^^^":  "^",
        "^^^":   "",
    })

    un_mille_million = pn.string_map({
        "un-mille": "mille",
        "un-millions": "un-million",
    })

    fixmeup = pn.string_map({
        # "zzzzz" : "xxxxxx",
        "-cent--millions--mille": "-cents-millions",
        "millions-un-mille": "millions-mille",
        # "million--mille": "million",
        "millions--mille": "millions",
        "vingts-mille": "vingt-mille",
        "cent--mille": "cent-mille",
    })

    fixmeup2 = pn.string_map({
        "-cent--millions": "-cents-millions",
        "--": "-",
    })

    decimals = pn.string_map({
        "0": "zéro ",  # zéro
        "1": "un ",
        "2": "deux ",
        "3": "trois ",
        "4": "quatre ",
        "5": "cinq ",
        "6": "six ",
        "7": "sept ",
        "8": "huit ",
        "9": "neuf ",

    })

    fsa_eos = pn.a("[EOS]")
    fsa_bos = pn.a("[BOS]")
    fsa_dot_comma = pn.u(".", ",")

    fst_dot_comma = pn.cdrewrite(pn.u(pn.t(".", " virgule "), pn.t(",", " virgule ")), "", "", alphabet_full)

    fst_decimals = pn.cdrewrite(decimals, "", "", alphabet_full)

    fst_zeros = pn.cdrewrite(zeros, "", fsa_0_9 | fsa_eos | fsa_dot_comma, alphabet_full);

    fst_single_zero = pn.cdrewrite(single_zero, "", fsa_eos | fsa_dot_comma, alphabet_full);

    fst_single_digits = pn.cdrewrite(single_digits, "", pn.u(fsa_eos, "-", fsa_dot_comma), alphabet_full)

    fst_teens = pn.cdrewrite(teens_10_19, "", "", alphabet_full)

    fst_mult_20_60 = pn.cdrewrite(mult_20_60, "", "", alphabet_full)
    fst_mult_2x_6x = pn.cdrewrite(mult_2x_6x, "", fsa_0_9, alphabet_full)

    fst_mult_70_90 = pn.cdrewrite(mult_70_90, "", "", alphabet_full)
    fst_mult_8x = pn.cdrewrite(mult_8x, "", fsa_0_9, alphabet_full)

    fst_hundreds_alone = pn.cdrewrite(hundreds_alone, "", fsa_eos, alphabet_full)
    fst_hundreds = pn.cdrewrite(hundreds, "", fsa_0_9, alphabet_full)

    fst_mille = pn.cdrewrite(mille, "", fsa_0_9, alphabet_full)
    fst_million = pn.cdrewrite(million, "", fsa_0_9, alphabet_full)

    fst_strip_triple_factor = pn.cdrewrite(strip_triple_factor, fsa_0_9, pn.u(" ", "-"), alphabet_full)

    fst_un_mille_million = pn.cdrewrite(un_mille_million, fsa_bos, "", alphabet_full)

    fst_fixmeup = pn.cdrewrite(fixmeup, "", "", alphabet_full)
    fst_fixmeup2 = pn.cdrewrite(fixmeup2, "", "", alphabet_full)

    fst = factor_fst * fst_million * fst_mille * fst_strip_triple_factor * \
        fst_hundreds_alone * fst_hundreds * \
        fst_mult_70_90 * fst_mult_8x * fst_mult_20_60 * fst_mult_2x_6x * \
        fst_teens * fst_zeros * fst_single_zero * fst_single_digits * \
        fst_un_mille_million * fst_fixmeup * fst_fixmeup2 * \
        fst_dot_comma * fst_decimals

    fst = fst.optimize()

    return fst


def generate_french_numerals(factor_fst):
    # full french alphabet - https://en.wikiversity.org/wiki/French/Alphabet
    alphabet_full = pn.u(*".0123456789^ _-abcdefghijklmnopqrstuvwxyzàèùéâêîôûëïüÿæœç").star
    fsa_0_9 = pn.u(*"0123456789").star

    single_zero = pn.t("0", "zéro")

    single_digits = pn.string_map({
        "0": "",  # zéro
        "1": "un",
        "2": "deux",
        "3": "trois",
        "4": "quatre",
        "5": "cinq",
        "6": "six",
        "7": "sept",
        "8": "huit",
        "9": "neuf",
    })

    zeros = pn.string_map({
        # "0^^ 0^ 0": "",
        "0^ ":  "",
        "0^^ ": "",
        "mille_0^^ 0^ 0":   "mille",
    })

    teens_10_19 = pn.string_map({
        "1^ 0": "dix",
        "1^ 1": "onze",
        "1^ 2": "douze",
        "1^ 3": "treize",
        "1^ 4": "quatorze",
        "1^ 5": "quinze",
        "1^ 6": "seize",
        "1^ 7": "dix-sept",
        "1^ 8": "dix-huit",
        "1^ 9": "dix-neuf",
    })

    mult_20_60 = pn.string_map({
        "2^ 0": "vingt",
        "2^ 1": "vingt_et_un",
        "3^ 0": "trente",
        "3^ 1": "trente_et_un",
        "4^ 0": "quarante",
        "4^ 1": "quarante_et_un",
        "5^ 0": "cinquante",
        "5^ 1": "cinquante_et_un",
        "6^ 0": "soixante",
        "6^ 1": "soixante_et_un",
    })

    mult_2x_6x = pn.string_map({
        "2^ ": "vingt-",
        "3^ ": "trente-",
        "4^ ": "quarante-",
        "5^ ": "cinquante-",
        "6^ ": "soixante-",
    })

    mult_70_90 = pn.string_map({
        "7^ 0": "soixante-dix",
        "7^ 1": "soixante_et_onze",
        "7^ 2": "soixante-douze",
        "7^ 3": "soixante-treize",
        "7^ 4": "soixante-quatorze",
        "7^ 5": "soixante-quinze",
        "7^ 6": "soixante-seize",
        "7^ 7": "soixante-dix-sept",
        "7^ 8": "soixante-dix-huit",
        "7^ 9": "soixante-dix-neuf",

        "8^ 0": "quatre-vingts",

        "9^ 0": "quatre-vingt-dix",
        "9^ 1": "quatre-vingt-onze",
        "9^ 2": "quatre-vingt-douze",
        "9^ 3": "quatre-vingt-treize",
        "9^ 4": "quatre-vingt-quatorze",
        "9^ 5": "quatre-vingt-quinze",
        "9^ 6": "quatre-vingt-seize",
        "9^ 7": "quatre-vingt-dix-sept",
        "9^ 8": "quatre-vingt-dix-huit",
        "9^ 9": "quatre-vingt-dix-neuf",
    })

    mult_8x = pn.string_map({
        "8^ ": "quatre-vingt-",
    })

    hundreds_alone = pn.string_map({
        "1^^ 0^ 0": "cent",
        "2^^ 0^ 0": "deux_cents",
        "3^^ 0^ 0": "trois_cents",
        "4^^ 0^ 0": "quatre_cents",
        "5^^ 0^ 0": "cinq_cents",
        "6^^ 0^ 0": "six_cents",
        "7^^ 0^ 0": "sept_cents",
        "8^^ 0^ 0": "huit_cents",
        "9^^ 0^ 0": "neuf_cents",
    })

    hundreds = pn.string_map({
        "1^^ ": "cent_",
        "2^^ ": "deux_cent_",
        "3^^ ": "trois_cent_",
        "4^^ ": "quatre_cent_",
        "5^^ ": "cinq_cent_",
        "6^^ ": "six_cent_",
        "7^^ ": "sept_cent_",
        "8^^ ": "huit_cent_",
        "9^^ ": "neuf_cent_",
    })

    mille = pn.string_map({
        "0^^^ ": "0^^^_mille_",
        "1^^^ ": "1^^^_mille_",
        "2^^^ ": "2^^^_mille_",
        "3^^^ ": "3^^^_mille_",
        "4^^^ ": "4^^^_mille_",
        "5^^^ ": "5^^^_mille_",
        "6^^^ ": "6^^^_mille_",
        "7^^^ ": "7^^^_mille_",
        "8^^^ ": "8^^^_mille_",
        "9^^^ ": "9^^^_mille_",
    })

    million = pn.string_map({
        "0^^^^^^ ": "0^^^^^^_millions_",
        "1^^^^^^ ": "1^^^^^^_millions_",
        "2^^^^^^ ": "2^^^^^^_millions_",
        "3^^^^^^ ": "3^^^^^^_millions_",
        "4^^^^^^ ": "4^^^^^^_millions_",
        "5^^^^^^ ": "5^^^^^^_millions_",
        "6^^^^^^ ": "6^^^^^^_millions_",
        "7^^^^^^ ": "7^^^^^^_millions_",
        "8^^^^^^ ": "8^^^^^^_millions_",
        "9^^^^^^ ": "9^^^^^^_millions_",
    })

    strip_triple_factor = pn.string_map({
        "^^^^^^^^": "^^",
        "^^^^^^^": "^",
        "^^^^^^": "",
        "^^^^^": "^^",
        "^^^^":  "^",
        "^^^":   "",
    })

    un_mille_million = pn.string_map({
        "un_mille": "mille",
        "un_millions": "un_million",
    })

    fixmeup = pn.string_map({
        # "zzzzz" : "xxxxxx",
        "_cent__millions__mille": "_cents_millions",
        "millions_un_mille": "millions_mille",
        # "million--mille": "million",
        "millions__mille": "millions",
        "vingts_mille": "vingt_mille",
        "cent__mille": "cent_mille",
    })

    fixmeup2 = pn.string_map({
        "million__mille": "million",
        "_cent__millions": "_cents_millions",
        "million_un_mille": "million_mille",
        "__": "_",
    })

    decimals = pn.string_map({
        "0": "zéro ",  # zéro
        "1": "un ",
        "2": "deux ",
        "3": "trois ",
        "4": "quatre ",
        "5": "cinq ",
        "6": "six ",
        "7": "sept ",
        "8": "huit ",
        "9": "neuf ",
        "_": " ",
    })

    fsa_eos = pn.a("[EOS]")
    fsa_bos = pn.a("[BOS]")
    fsa_dot_comma = pn.u(".", ",")

    fst_dot_comma = pn.cdrewrite(pn.u(pn.t(".", " virgule "), pn.t(",", " virgule ")), "", "", alphabet_full)

    fst_decimals = pn.cdrewrite(decimals, "", "", alphabet_full)

    fst_zeros = pn.cdrewrite(zeros, "", fsa_0_9 | fsa_eos | fsa_dot_comma, alphabet_full);

    fst_single_zero = pn.cdrewrite(single_zero, "", fsa_eos | fsa_dot_comma, alphabet_full);

    fst_single_digits = pn.cdrewrite(single_digits, "", pn.u(fsa_eos, "-", "_", fsa_dot_comma), alphabet_full)

    fst_teens = pn.cdrewrite(teens_10_19, "", "", alphabet_full)

    fst_mult_20_60 = pn.cdrewrite(mult_20_60, "", "", alphabet_full)
    fst_mult_2x_6x = pn.cdrewrite(mult_2x_6x, "", fsa_0_9, alphabet_full)

    fst_mult_70_90 = pn.cdrewrite(mult_70_90, "", "", alphabet_full)
    fst_mult_8x = pn.cdrewrite(mult_8x, "", fsa_0_9, alphabet_full)

    fst_hundreds_alone = pn.cdrewrite(hundreds_alone, "", fsa_eos, alphabet_full)
    fst_hundreds = pn.cdrewrite(hundreds, "", fsa_0_9, alphabet_full)

    fst_mille = pn.cdrewrite(mille, "", fsa_0_9, alphabet_full)
    fst_million = pn.cdrewrite(million, "", fsa_0_9, alphabet_full)

    fst_strip_triple_factor = pn.cdrewrite(strip_triple_factor, fsa_0_9, pn.u(" ", "-", "_"), alphabet_full)

    fst_un_mille_million = pn.cdrewrite(un_mille_million, fsa_bos, "", alphabet_full)

    fst_fixmeup = pn.cdrewrite(fixmeup, "", "", alphabet_full)
    fst_fixmeup2 = pn.cdrewrite(fixmeup2, "", "", alphabet_full)

    fst = factor_fst * fst_million * fst_mille * fst_strip_triple_factor * \
        fst_hundreds_alone * fst_hundreds * \
        fst_mult_70_90 * fst_mult_8x * fst_mult_20_60 * fst_mult_2x_6x * \
        fst_teens * fst_zeros * fst_single_zero * fst_single_digits * \
        fst_un_mille_million * fst_fixmeup * fst_fixmeup2 * \
        fst_dot_comma * fst_decimals

    fst = fst.optimize()

    return fst


def test_output(num, fst_factor, fst_numeral, hyphens=False):
    print(num)
    print((str(num) * fst_factor).stringify())
    print((str(num) * fst_numeral).stringify())

    snum = num2words(num, lang='fr')
    if hyphens:
        snum = snum.replace().replace(' ', '-')

    print(snum)
    print()

    return


def compare_output(num_start, num_end, fst_factor, fst_numeral, hyphens=False):
    error_count = 0
    for num in range(num_start, num_end + 1):
        a = (str(num) * fst_numeral).stringify()

        b = num2words(num, lang='fr')
        if hyphens:
            b = b.replace(' ', '-')

        if a != b:
            error_count += 1
            print(num, (str(num) * fst_factor).stringify())
            print(a)
            print(b)
            print()

    # print("Comparing from", num_start, "to:", num_end, "ERRORS:", error_count)
    return


print("---------- main -------------")
fst_factor = generate_fst_digit()
# fst_numeral = generate_french_numerals_hyphens(fst_factor)
fst_numeral = generate_french_numerals(fst_factor)

# for num in range(0, 101):
#     print(num, (str(num) * fst_factor).stringify(), (str(num) * fst_numeral).stringify())
#

#x=1
#for num in range(1,10):
#    test_output(x, fst_factor, fst_numeral)
#    x = x * 10

#
# num = 121
# test_output(num, fst_factor, fst_numeral)
# num = 121121
# test_output(num, fst_factor, fst_numeral)
# num = 121121121
# test_output(num, fst_factor, fst_numeral)
#
# num = 286980688
# test_output(num, fst_factor, fst_numeral)
# num = 180980280
# test_output(num, fst_factor, fst_numeral)
# num = 200200200
# test_output(num, fst_factor, fst_numeral)
# num = 100100100
# test_output(num, fst_factor, fst_numeral)
#
# num = 700700
# test_output(num, fst_factor, fst_numeral)
# num = 710710
# test_output(num, fst_factor, fst_numeral)
# num = 786786
# test_output(num, fst_factor, fst_numeral)
#
# num = 999999
# test_output(num, fst_factor, fst_numeral)
# num = 998999
# test_output(num, fst_factor, fst_numeral)

# num = 999999999
# test_output(num, fst_factor, fst_numeral)

#0.999, 0.0110, 1.01, 101.01,
nums = [0.999, 123.004, 9051631, 9061165,  9231720, 1001628, 1001369, 1010369]
# , 800582627, 700324222, 300221861, 900098716, 884001000, 300297420, 300056341, 252001347, 400286533, 900489719, 1515252, 824000424, 200579898

for num in nums:
    # print(("1.01"*fst_factor).stringify())
    # test_output(num, fst_factor, fst_numeral)
    compare_output(num, num, fst_factor, fst_numeral)

for x in range(0, 1000000+1):
    num = random.randint(0, 999999999)
    if x % 1000 == 0:
        print(x, num)
    # num = random.randint(0, 999999999)
    compare_output(num, num, fst_factor, fst_numeral)
    #test_output(num, fst_factor, fst_numeral)

print("---------- done -------------")
