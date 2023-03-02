


""" Module for badges colors """

from colorsys import hsv_to_rgb

def interpolate(weight, x, y):
    return x * weight + (1-weight) * y

def score_to_rgb_color(score, score_min, score_max):
    normalized_score = max(0, (score - score_min) / (score_max - score_min))
    hsv_color = (interpolate(normalized_score, 0.33, 0), 1, 1)
    rgb_color = hsv_to_rgb(*hsv_color)
    rgb_color = tuple(int(255*value) for value in rgb_color)
    return f"rgb{rgb_color}"
