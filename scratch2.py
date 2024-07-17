def levenshtein_two_matrix_rows(str1, str2):
    # Get the lengths of the input strings
    m = len(str1)
    n = len(str2)

    # Initialize two rows for dynamic programming
    prev_row = [j for j in range(n + 1)]
    curr_row = [0] * (n + 1)

    # Dynamic programming to fill the matrix
    for i in range(1, m + 1):
        # Initialize the first element of the current row
        curr_row[0] = i

        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, no operation needed
                curr_row[j] = prev_row[j - 1]
            else:
                # Choose the minimum cost operation
                curr_row[j] = 1 + min(
                    curr_row[j - 1],  # Insert
                    prev_row[j],  # Remove
                    prev_row[j - 1]  # Replace
                )

        # Update the previous row with the current row
        prev_row = curr_row.copy()

    # The final element in the last row contains the Levenshtein distance
    return curr_row[n]


# Driver code
if __name__ == "__main__":
    # Example input strings
    str1 = "kitten"
    str2 = "sitting"

    # Function call to calculate Levenshtein distance
    distance = levenshtein_two_matrix_rows(str1, str2)

    # Print the result
    print("Levenshtein Distance:", distance)
