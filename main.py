# main.py
from anpr import ANPRSystem

if __name__ == "__main__":
    anpr = ANPRSystem()
    anpr.process_video(show_video=False)  # # Set to False to disable Live