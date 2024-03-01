import numpy as np
import math


def score_function(A, B):
    """
    A and B are pitch-class representations
    :param A:
    :param B:
    :return:
    """
    denominator = len(A | B)
    if denominator == 0:
        # Means both are silence
        return 1
    AandB = A & B
    AorB = A | B
    posTerm = len(AandB)
    negTerm = len(AorB - AandB)
    score = (posTerm - negTerm) / denominator
    return 3 * score


# def score_function(orchestra, piano):
#     score = 0
#     for (pitch_b, type_b) in piano:
#         for pitch_a, type_a in orchestra:
#             #  Same notes
#             if pitch_a == pitch_b:
#                 if
#             number_same_notes = len([_ for e in B if (e[0] == pitch) and (e[1] == type)])
#             number_same_notes_diff_type = len([_ for e in B if (e[0] == pitch) and (e[1] == type)])
#             # Octave
#             number_same_notes = len([_ for e in B if (e[0] == pitch) and (e[1] == type)])
#             number_same_notes_diff_type = len([_ for e in B if (e[0] == pitch) and (e[1] == type)])
#
#             #  Different note
#
#     return


def nwalign(seqj, seqi, gapOpen, gapExtend, score_matrix):
    """
    >>> global_align('COELANCANTH', 'PELICAN')
    ('COELANCANTH', '-PEL-ICAN--')
    nwalign must be used on list of pitch_class ensemble
    Two versions of the sequences are returned
    First tuple is with removed elements (useful for training and initialize generation)
    Second tuple simply indicates where elemnts have been skipped in each sequence (useful for generating after a sequence has been init)
    TODO:
    Limit search zone
    Abort exploration if score becomes too small
    """

    UP, LEFT, DIAG, NONE = range(4)

    max_j = len(seqj)
    max_i = len(seqi)

    score = np.zeros((max_i + 1, max_j + 1), dtype='f') - math.inf
    pointer = np.zeros((max_i + 1, max_j + 1), dtype='i')
    max_i, max_j

    pointer[0, 0] = NONE
    score[0, 0] = 0.0

    pointer[0, 1:] = LEFT
    pointer[1:, 0] = UP

    # Do we do that ?? Not sure...
    score[0, 1:] = gapExtend * np.arange(max_j)
    score[1:, 0] = gapExtend * np.arange(max_i)

    termScores = []

    ##################################
    #  Build score matrix
    for i in range(1, max_i + 1):
        ci = seqi[i - 1]

        # Constraint to a losange
        # Faster, probably sufficient for almost aligned sequences like in Beethov/Liszt,
        # but probably do not work in general
        j_min = max(i - 1000, 1)
        j_max = min(i + 1000, max_j + 1)
        for j in range(j_min, j_max):
            # for j in range(1, max_j + 1):
            cj = seqj[j - 1]

            if score_matrix is not None:
                termScore = score_matrix[ci, cj]
            else:
                termScore = score_function(ci, cj)
            termScores.append(termScore)
            diag_score = score[i - 1, j - 1] + termScore

            if pointer[i - 1, j] == UP:
                up_score = score[i - 1, j] + gapExtend
            else:
                up_score = score[i - 1, j] + gapOpen

            if pointer[i, j - 1] == LEFT:
                left_score = score[i, j - 1] + gapExtend
            else:
                left_score = score[i, j - 1] + gapOpen

            if diag_score >= up_score:
                if diag_score >= left_score:
                    score[i, j] = diag_score
                    pointer[i, j] = DIAG
                else:
                    score[i, j] = left_score
                    pointer[i, j] = LEFT

            else:
                if up_score > left_score:
                    score[i, j] = up_score
                    pointer[i, j] = UP
                else:
                    score[i, j] = left_score
                    pointer[i, j] = LEFT

    ##################################
    #  Build aligned indices
    pairs = []
    previous_coord = None
    while True:

        p = pointer[i, j]

        if p == NONE:
            break

        if p == DIAG:
            i -= 1
            j -= 1
            pairs.append((j, i))
            # if previous_coord is not None:
            #     pairs.append(previous_coord)
        elif p == LEFT:
            j -= 1
        elif p == UP:
            i -= 1
        else:
            raise Exception('wtf!')

        # if (i != len(seqi)) and (j != len(seqj)):
        #     previous_coord = j, i

    # Don't forget last one
    # pairs.append(previous_coord)

    # return (align_j[::-1], align_i[::-1]), (skip_j[::-1], skip_i[::-1])
    return pairs[::-1], score
