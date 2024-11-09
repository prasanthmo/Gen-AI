import random

# Total number of cards in a deck
total_cards = 52
black_kings = 2
black_cards = 26  # 13 clubs + 13 spades (two of these are black kings)

# P(A): Probability of picking a black king
p_black_king = black_kings / total_cards

# P(B): Probability of picking a black card
p_black_card = black_cards / total_cards

# P(B|A): Probability of picking a black card given that it was a black king (which is always true)
p_black_card_given_black_king = 1

# P(A|B): Probability of picking a black king given that we picked a black card
p_black_king_given_black_card = (p_black_card_given_black_king * p_black_king) / p_black_card

print(f"Probability of picking a black king given that the card is black: {p_black_king_given_black_card:.4f}")
