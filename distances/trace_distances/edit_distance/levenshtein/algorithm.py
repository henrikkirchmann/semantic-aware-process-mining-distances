def compute_levenshtein_distance(trace1, trace2, substitution_scores):
    m = len(trace1)
    n = len(trace2)
    substitution_scores = substitution_scores[next(iter(substitution_scores))]

    # Initialize a matrix to store the edit distances
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize the first row and column with values from 0 to m and 0 to n respectively
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix using dynamic programming to compute edit distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if trace1[i - 1] == trace2[j - 1]:
                # Characters match, no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Characters don't match, choose minimum cost among insertion, deletion, or substitution
                dp[i][j] = min(
                    1 + dp[i][j - 1], #insertion
                    1 + dp[i - 1][j], #deletion
                    substitution_scores[(trace1[i - 1], trace2[j - 1])] + dp[i - 1][j - 1] #substitution
                )

    # Return the edit distance between the strings
    return dp[m][n]


def get_levenshtein_distance(trace1, trace2, substitution_scores):
    if next(iter(substitution_scores))  == "Bose 2009 Substitution Scores": #turn normalized simalrity into distance
        for key in substitution_scores["Bose 2009 Substitution Scores"].keys():
            substitution_scores["Bose 2009 Substitution Scores"][key] = 1 - substitution_scores["Bose 2009 Substitution Scores"][key]
    return compute_levenshtein_distance(trace1, trace2, substitution_scores)


