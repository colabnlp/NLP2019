import pynini as pn
import pytest


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


@pytest.fixture(scope="module")
def n2w_fst():
    factor_fst = generate_fst_digit()

    # full french alphabet - https://en.wikiversity.org/wiki/French/Alphabet
    alphabet_full = pn.u(*".0123456789^ _-abcdefghijklmnopqrstuvwxyzàèùéâêîôûëïüÿæœç").star
    fsa_0_9 = pn.u(*"0123456789").star

    # single_zero = pn.t("0", "zéro")
    single_zero = pn.t("0", "zero")

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
        # "0": "zéro ",  # zéro
        "0": "zero ",  # zéro
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

    transformer = fst.optimize()



    ## ---------- YOUR PART ENDS------------
    return transformer


def n2w(fst, number_as_string):
    return (number_as_string * fst).stringify()


@pytest.mark.parametrize("test_input,expected", [
    ("1", "un"),
    ("0", "zero"),
    ("10",  "dix"),
    ("21",  "vingt et un"),
    ("10",  "dix"),
    ("30",  "trente"),
    ("21",  "vingt et un"),
    ("45",  "quarante-cinq"),
    ("99",  "quatre-vingt-dix-neuf"),
    ("100",  "cent"),
    ("110",  "cent dix"),
    ("121",  "cent vingt et un"),
    ("45",  "quarante-cinq"),
    ("99",  "quatre-vingt-dix-neuf"),
    ("110",  "cent dix"),
    ("121",  "cent vingt et un"),
    ("100000", "cent mille"),
    ("1000010", "un million dix"),
    ("1001628", "un million mille six cent vingt-huit"),
    ("0.46", "zero virgule quatre six"),
    ("0.046", "zero virgule zero quatre six"),
    ]) 
def test_numbers(n2w_fst, test_input, expected):
  assert n2w(n2w_fst, test_input) == expected


