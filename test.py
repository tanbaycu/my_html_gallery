import random
import time
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def color(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def blink(text):
    return f"\033[5m{text}\033[0m"

def rainbow_text(text):
    colors = [91, 93, 92, 96, 94, 95]
    return ''.join(color(char, random.choice(colors)) for char in text)

def print_spectacular_christmas_tree(height):
    ornaments = ['●', '○', '✦', '✧', '♦', '♥', '♠', '♣', '❄', '✴', '✳', '⚝']
    star = blink(color('★', 93))
    
    for frame in range(5):  # 5 frames of animation
        clear_screen()
        
        # Print star
        print(' ' * (height - 1) + star)
        
        # Print tree
        for i in range(1, height + 1):
            spaces = ' ' * (height - i)
            tree = ''
            for j in range(2 * i - 1):
                if random.random() < 0.3:  # 30% chance for an ornament
                    ornament = random.choice(ornaments)
                    tree += blink(color(ornament, random.randint(91, 96)))
                else:
                    tree += color('♣', 32)  # Green tree
            print(spaces + tree)
            
            # Add garland
            if i > 1:
                garland = ''.join(color('~', random.randint(91, 96)) for _ in range(2 * i - 3))
                print(' ' * (height - i + 1) + garland)
        
        # Print trunk
        trunk_width = 5
        trunk_height = 4
        for _ in range(trunk_height):
            print(' ' * (height - trunk_width // 2) + color('|' * trunk_width, 33))
        
        # Print base
        base_width = height * 2 - 1
        print(color('▀' * base_width, 33))
        
        # Print gifts
        gifts = [color('■', c) for c in [91, 92, 93, 94, 95, 96]]
        print(' '.join(random.choices(gifts, k=base_width // 2)))
        
        # Print message
        message = "CHÚC MỪNG GIÁNG SINH!"
        print('\n' + ' ' * ((base_width - len(message)) // 2) + rainbow_text(message))
        
        # Snowfall
        for _ in range(height):
            snow_line = ' ' * random.randint(0, base_width - 1) + color('*', 97)
            snow_line += ' ' * (base_width - len(snow_line))
            print(snow_line)
        
        time.sleep(0.5)  # Pause between frames

# Tree height
tree_height = 15

print("Cây thông Noel cực kỳ đẹp và hoành tráng:")
for _ in range(10):  # Run animation 10 times
    print_spectacular_christmas_tree(tree_height)

