import tkinter as tk
import json 
import gurobipy as gp
import numpy as np

class Menu:
    def menu_command(self, command):
        print(command["label"])

    # commands: 
    # [{"label": "Annual Report", "frequency": 0.6}]
    def __init__(self, menu_items):
        self.root = tk.Tk()
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        for command in menu_items:
            filemenu.add_command(label=command["label"], command=lambda cmd=command: self.menu_command(cmd))
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)
        self.root.mainloop()

def main():
    commands = json.load(open("commands.json"))
    menu = Menu(commands)

if __name__ == "__main__":
    main()