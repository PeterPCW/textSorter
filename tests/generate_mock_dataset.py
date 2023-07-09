import random
import string

def generate_mock_dataset(size, repeat_frequency):
    lines = []
    for _ in range(size):
        line = generate_random_line()
        lines.append(line)

    # Introduce similarity between lines at the word level
    for i in range(size):
        for j in range(i+1, size):
            words_i = lines[i].split()
            words_j = lines[j].split()
            for k in range(min(len(words_i), len(words_j))):
                if random.random() < repeat_frequency:
                    words_j[k] = words_i[k]

            lines[j] = ' '.join(words_j)

    return '\n'.join(lines)

def generate_random_line():
    line_length = random.randint(5, 15)
    line = ' '.join(''.join(random.choices(string.ascii_letters, k=random.randint(3, 8))) for _ in range(line_length))
    return line