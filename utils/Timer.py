def convert_time(seconds):
    return f"{int(seconds // 3600)}h, {int(seconds//60)}m, {seconds%60:.2f}"